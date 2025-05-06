#include "llama.cuh"
#include "utils.cuh"
using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;
    using config = default_config;

    template <typename Config, typename Globals>
    struct rms_upgate_silu
    {
        static constexpr int opcode = OPCODE_RMS_DoubleMatVecSiLU; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int prev_opcode = OPCODE_O_ProjResidual;
        static constexpr int EXPECTED_ARRIVAL_COUNT = Globals::hidden_dim / Globals::matvec_block_size;

        struct parsed_instruction
        {
            int layer_idx, start_block_idx, end_block_idx, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];
                start_block_idx = instruction[2];
                end_block_idx = instruction[3];
                iters = 2 * (end_block_idx - start_block_idx);
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics
        {
            static __device__ inline void gmem_wait(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                while (*(volatile int *)&g.Bar[{inst.layer_idx, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
            }

            static __device__ inline void load_iter(state<Config> &s, const globals &g, parsed_instruction &inst, int iter, int col_idx, st_bf<16, 512> &weight_chunk, semaphore &sem)
            {
                auto block_idx = inst.start_block_idx + iter / 2;
                if (iter % 2 == 0)
                {
                    tma::load_async(weight_chunk, g.up_weights, {inst.layer_idx, block_idx, col_idx}, sem);
                }
                else
                {
                    tma::load_async(weight_chunk, g.gate_weights, {inst.layer_idx, block_idx, col_idx}, sem);
                }
            }

            static __device__ inline void store(state<Config> &s, const Globals &g, parsed_instruction &inst, int output_idx, int output_stage, semaphore &sem, int bit)
            {
                // mbarriers need to be waited on for every phase.
                wait(sem, bit);

                if (output_idx % 2 == 0)
                {
                    return;
                }

                auto true_output_idx = output_idx / 2;

                // NOTE: hardcoding to 3 output stages for now
                auto prev_output_idx = (output_idx - 1);
                auto prev_output_stage = prev_output_idx % 3;

                int block_idx = inst.start_block_idx + true_output_idx;

                sv_fl<16> &up_out_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * prev_output_stage));
                sv_fl<16> &gate_out_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));

                sv_bf<16> &out_smem = *reinterpret_cast<sv_bf<16> *>(&gate_out_smem);

                rv_fl<16> up_out, gate_out, gate_scratch;

                warp::load(up_out, up_out_smem);
                warp::load(gate_out, gate_out_smem);

                // neg
                warp::mul(gate_scratch, gate_out, -1.f);
                warp::exp(gate_scratch, gate_scratch);
                warp::add(gate_scratch, gate_scratch, 1.f);
                warp::div(gate_out, gate_out, gate_scratch);

                // gating
                warp::mul(gate_out, up_out, gate_out);

                // wait before we overwrite gate_out
                warp::sync();

                warp::store(out_smem, gate_out);

                // wait before we store results to global memory
                warp::sync();

                if (laneid() == 0)
                {
                    tma::store_async<cache_policy::NORMAL>(g.silu_out, out_smem, {block_idx});
                    tma::store_async_read_wait();
                }

                warp::sync();
                warp::zero(up_out_smem);
                warp::zero(gate_out_smem);
                warp::sync();
            }
        };

        using pipeline = rms_matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;
        static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3);

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                return pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                return pipeline::init_semaphores(s);
            }
        };

        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                s.template zero_scratch<1024>();

                parsed_instruction inst{s};
                pipeline::loader_loop<&Globals::mlp_norm_weights>(s, g, inst.layer_idx);
            }
        };

        struct launcher
        {
            // launcher does nothing here, since this doesn't use tensor cores.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                pipeline::launcher_loop<&Globals::hidden_states>(s, g);
            }
        };

        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::consumer_loop(s, g);
            }
        };

        struct storer
        {

            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::storer_loop<2>(s, g);

                // one atomic add at the end
                if (laneid() == 0)
                {
                    tma::store_async_wait();
                    asm volatile("fence.acq_rel.gpu;");

                    parsed_instruction inst{s};
                    auto to_increment = inst.end_block_idx - inst.start_block_idx;
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}], to_increment);
                }
            }
        };
    };
}
