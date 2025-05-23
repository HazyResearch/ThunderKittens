#pragma once

#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    template <
        int _EXPECTED_ARRIVAL_COUNT,
        auto WeightsPtr,
        auto InputActivationsPtr,
        auto OutputActivationsPtr,
        int _opcode,
        int _prev_opcode = 0,
        typename Config = kittens::prototype::vm::default_config,
        typename Globals = llama_1b_globals>

    struct MatVecAddOp
    {
        static constexpr int opcode = _opcode;
        static constexpr int prev_opcode = _prev_opcode;
        static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;

        struct parsed_instruction
        {
            int layer, start_block_idx, end_block_idx, reduction_block_idx, start_reduction_col, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];               // in units of 1
                start_block_idx = instruction[2];     // in units of 1 (0, 16, 32, ..., 2032)
                end_block_idx = instruction[3];       // in units of 1 (0, 16, 32, ..., 2032)
                reduction_block_idx = instruction[4]; // in units of hidden_dim=2048 (0, 2048, 4096, 6144)
                start_reduction_col = reduction_block_idx * Globals::hidden_dim;
                iters = end_block_idx - start_block_idx;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics
        {

            static __device__ inline void load_iter(state<Config> &s, const globals &g, parsed_instruction &inst, int iter, int col_idx, st_bf<16, 512> &weight_chunk, semaphore &sem)
            {
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(weight_chunk, g.*WeightsPtr, coord<>{inst.layer, (inst.start_block_idx + iter) * Globals::matvec_block_size, inst.start_reduction_col + 512 * col_idx}, sem);
            }

            static __device__ inline void store(state<Config> &s, const globals &g, parsed_instruction &inst, int output_idx, int output_stage)
            {

                int block_idx = inst.start_block_idx + output_idx;

                sv_fl<16> &output_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));
                sv_bf<16> &output_smem_bf = *reinterpret_cast<sv_bf<16> *>((float *)s.scratch() + (32 * output_stage));

                rv_fl<16> output_rv;
                warp::load(output_rv, output_smem);
                warp::sync();
                warp::store(output_smem_bf, output_rv);
                warp::sync();

                if (warp::laneid() == 0)
                {
                    auto &OutputActivations = g.*OutputActivationsPtr; // object in global memory
                    tma::store_add_async<cache_policy::EVICT_LAST>(OutputActivations, output_smem_bf, {block_idx});
                    tma::store_async_read_wait();
                }

                warp::sync();
                warp::zero(output_smem);
                warp::sync();
            }
        };
        using pipeline = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;

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

                pipeline::loader_loop(s, g);
            }
        };
        struct launcher
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (laneid() == 0)
                {
#ifdef KITTENS_BLACKWELL
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                using sv_t = sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
                using rv_t = rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
                parsed_instruction inst{s};

                if (laneid() == 0 && warpid() == 0)
                {

                    int activation_page = pipeline::get_activation_page(s);
                    s.wait_page_ready(activation_page);

                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                    {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                    s.record(TEVENT_DONE_GMEM_WAIT);

                    auto &activations = pipeline::get_activations(s);
                    auto &InputActivations = g.*InputActivationsPtr; // object in global memory
                }
                group<Config::NUM_CONSUMER_WARPS>::sync(4);

                sv_t &activations_smem = reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[warpid()];

                warp::load(activations_smem, g.*InputActivationsPtr, coord<>{inst.start_reduction_col + warpid() * pipeline::REDUCTION_DIM_PER_WARP});
                warp::sync();

                rv_t activations_vec;
                warp::load(activations_vec, activations_smem);
                warp::sync();

                s.warp_finish_page(pipeline::get_activation_page(s), 1);

                pipeline::consumer_loop(s, g, activations_vec);
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                pipeline::storer_loop(s, g);
                warp::sync();

                if (laneid() == 0)
                {
                    s.record(TEVENT_AT_GMEM_STORE);
                    parsed_instruction inst{s};

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!

                    // asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);
                    s.record(TEVENT_DONE_GMEM_STORE);
                }
            }
        };
    };

    template <typename Config, typename Globals>
    struct downproj : MatVecAddOp<
                          llama_1b_globals::intermediate_dim / llama_1b_globals::matvec_block_size,
                          &Globals::down_weights,
                          &Globals::silu_out,
                          &Globals::hidden_states,
                          OPCODE_DownProjResidual,
                          OPCODE_DownProjResidual - 1,
                          Config,
                          Globals>
    {
    };

    template <typename Config, typename Globals>
    struct o_proj : MatVecAddOp<
                        llama_1b_globals::num_attention_heads,
                        &Globals::o_weights,
                        &Globals::attn_out,
                        &Globals::hidden_states,
                        OPCODE_O_ProjResidual,
                        OPCODE_O_ProjResidual - 1,
                        Config,
                        Globals>
    {
    };
}
