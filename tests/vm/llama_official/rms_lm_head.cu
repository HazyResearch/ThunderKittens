#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_lm_head
    {
        static constexpr int opcode = OPCODE_RMS_LM_Head; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int EXPECTED_ARRIVAL_COUNT = 512;

        struct parsed_instruction
        {
            int start_block_idx, end_block_idx, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                start_block_idx = instruction[1];
                end_block_idx = instruction[2];
                iters = end_block_idx - start_block_idx;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics
        {
            static __device__ inline void gmem_wait(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                while (*(volatile int *)&g.Bar[{globals::num_layers - 1, OPCODE_DownProjResidual - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
            }

            static __device__ inline void load_iter(state<Config> &s, const globals &g, parsed_instruction &inst, int iter, int col_idx, st_bf<16, 512> &weight_chunk, semaphore &sem)
            {
                auto block_idx = inst.start_block_idx + iter;
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(weight_chunk, g.lm_head_weights, {block_idx, col_idx}, sem);
            }

            static __device__ inline void store(state<Config> &s, const Globals &g, parsed_instruction &inst, int output_idx, int output_stage)
            {

                int block_idx = inst.start_block_idx + output_idx;

                uint8_t *output_scratch_start = pipeline::get_output_start(s, output_stage);
                sv_bf<16> &logits_smem_bf = *reinterpret_cast<sv_bf<16> *>(output_scratch_start);

                rv_fl<16> logits_rv;
                matvec_reduce<Config, sv_fl<16>, rv_fl<16>, pipeline::SCRATCH_BYTES_PER_WARP>(output_scratch_start, logits_rv);
                
                warp::sync();
                warp::store(logits_smem_bf, logits_rv);
                warp::sync();

                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_OUTPUT_READY);

                    tma::store_async<cache_policy::EVICT_LAST>(g.logits, logits_smem_bf, {0, 0, 0, block_idx});
                    tma::store_async_read_wait();
                }

                warp::sync();
            }
        };

        using pipeline = rms_matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics, &Globals::hidden_states, &Globals::lm_head_norm_weights>;

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
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                s.template zero_scratch<1024>();
                pipeline::loader_loop(s, g, 0);
            }
        };
        struct launcher
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::launcher_loop(s, g);
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
                pipeline::storer_loop(s, g);
            }
        };
    };
}
