#pragma once

#include "llama.cuh"
#include "utils.cuh"

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
        typename Config = kittens::prototype::vm::default_config>

    struct MatVecAddOp
    {
        static constexpr int opcode = _opcode;
        static constexpr int prev_opcode = _prev_opcode;
        static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;

        static constexpr int NUM_WEIGHT_PAGES = 4;
        static constexpr int PAGE_WEIGHT_START = 0;
        static constexpr int PAGE_ACTIVATION = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;
        static constexpr int PAGE_COUNT = PAGE_ACTIVATION + 1; // 5

        static constexpr int REDUCTION_DIM_PER_WARP = llama_1b_globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        struct parsed_instruction
        {
            int layer, output_block_idx, reduction_block_idx, start_output_col, start_reduction_col;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];               // in units of 1
                output_block_idx = instruction[2];    // in units of 1 (0, 16, 32, ..., 2032)
                reduction_block_idx = instruction[3]; // in units of 2048 (0, 2048, 4096, 6144)
                start_output_col = output_block_idx * llama_1b_globals::matvec_block_size;
                start_reduction_col = reduction_block_idx * llama_1b_globals::hidden_dim;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };
        static __device__ inline parsed_instruction parse_instruction(const globals &g, state<Config> &s)
        {
            return parsed_instruction{s.instruction()[1], s.instruction()[2]};
        }
        __device__ static inline kittens::semaphore &inputs_arrived(state<Config> &s, int id)
        {
            return s.semaphores()[id];
        }
        __device__ static inline kittens::semaphore &outputs_arrived(state<Config> &s)
        {
            return s.semaphores()[4];
        }
        __device__ static inline kittens::semaphore &activations_arrived(state<Config> &s)
        {
            return s.semaphores()[5];
        }
        __device__ static inline int get_weight_page(state<Config> &s, int offset)
        {
            return s.pid(offset + 1);
        }
        __device__ static inline int get_activation_page(state<Config> &s)
        {
            return s.pid(0);
        }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename Config::instruction_t &instruction, int &query)
            {
                int ret_order[] = {5, 6, 7, 8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const globals &g, state<Config> &s)
            {
                for (int i = 0; i < 4; i++)
                {
                    kittens::init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
                }
                kittens::init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS); // outputs arrived.
                kittens::init_semaphore(activations_arrived(s), 1);
                return 6;
            }
        };
        struct loader
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[kittens::laneid()] = 0;
                kittens::warp::sync(); // done, now we can proceed to other things.

                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_START);

                    for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
                    {
                        s.wait_page_ready(get_weight_page(s, i));
                        auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(s.pages[get_weight_page(s, i)]);
                        s.record(TEVENT_TRIPLES_START + i);
                        kittens::tma::expect(inputs_arrived(s, i), weight_chunk);

                        auto &weights_global = g.*WeightsPtr; // object in global memory
                        kittens::tma::load_async(weight_chunk, weights_global, coord<>{inst.layer, inst.start_output_col, inst.start_reduction_col + 512 * i}, inputs_arrived(s, i));

                        // auto& weights_global = g.*WeightsPtr;      // object in global memory
                        // kittens::tma::load_async(weight_chunk, weights_global, coord<>{inst.layer, inst.start_output_col, inst.start_reduction_col + 512 * laneid()}, inputs_arrived(s, laneid()));
                    }

                    int activation_page = get_activation_page(s);
                    s.wait_page_ready(activation_page);

                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                        __nanosleep(20);
                    s.record(TEVENT_DONE_GMEM_WAIT);

                    auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[activation_page]);
                    s.record(TEVENT_TRIPLES_START + 4);
                    kittens::tma::expect(activations_arrived(s), activations);

                    auto &InputActivations = g.*InputActivationsPtr; // object in global memory
                    kittens::tma::load_async(activations, InputActivations, coord<>{inst.start_reduction_col}, activations_arrived(s));
                }
                else if (kittens::laneid() >= 5 && kittens::laneid() <= 12)
                {
                    int unused_page = s.pid(kittens::laneid());
                    s.wait_page_ready(unused_page);
                    kittens::arrive(s.page_finished[unused_page], Config::NUM_CONSUMER_WARPS); // Release the unused pages immediately.
                }

                warp::sync();
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_END);
                }
            }
        };
        struct launcher
        { // launches mma's
            // launcher does nothing here, since this doesn't use tensor cores.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                s.wait_tensor_ready();
                kittens::warp::arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
        };
        struct consumer
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {

                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + warpid());
                }

                using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using float_rv_t = rv_fl<16>;

                typename float_rt_t::row_vec activations_vec;

                static_assert(Config::NUM_CONSUMER_WARPS % NUM_WEIGHT_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by NUM_WEIGHT_PAGES");
                constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / NUM_WEIGHT_PAGES;

                int page_index = warpid() / WARPS_PER_PAGE;

                using sv_slice_t = sv_bf<REDUCTION_DIM_PER_WARP>;

                wait(activations_arrived(s), 0);

                int activation_page = get_activation_page(s);
                sv_slice_t(&activations_smem)[Config::NUM_CONSUMER_WARPS] = reinterpret_cast<sv_slice_t(&)[Config::NUM_CONSUMER_WARPS]>(s.pages[activation_page]);

                warp::load(activations_vec, activations_smem[warpid()]);
                warp::sync();
                warp::arrive(s.page_finished[activation_page]);

                matvec<float_rt_t, NUM_WEIGHT_PAGES>(g, s, activations_vec, inputs_arrived(s, page_index), get_weight_page(s, page_index), 0);

                warp::sync();
                warp::arrive(outputs_arrived(s));

                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_END + warpid());
                }
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                void *scratch = s.scratch();

                // Convert to bf and put back in shared memory
                sv_fl<16> &output = *reinterpret_cast<sv_fl<16> *>(scratch);
                sv_bf<16> &output_bf = *reinterpret_cast<sv_bf<16> *>(scratch);

                rv_bf<16> output_reg_bf;

                wait(outputs_arrived(s), 0);
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_OUTPUT_READY);
                }

                warp::load(output_reg_bf, output);
                warp::sync();
                warp::store(output_bf, output_reg_bf);
                warp::sync();

                if (laneid() == 0)
                {
                    auto &OutputActivations = g.*OutputActivationsPtr; // object in global memory
                    kittens::tma::store_add_async(OutputActivations, output_bf, {inst.output_block_idx});
                    kittens::tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                }
                kittens::warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                if (laneid() == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], 1);
                    // if constexpr (OP_IDX == g.Bar.rows() - 1)
                    //     atomicAdd(&g.Bar[{inst.layer + 1, 0, 0}], 1);
                    // else
                    //     atomicAdd(&g.Bar[{inst.layer, OP_IDX + 1, 0}], 1);
                }

                warp::sync();
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_STORE_END);
                }
            }
        };
    };

    template <typename Config, typename Globals>
    struct downproj : MatVecAddOp<
                          llama_1b_globals::intermediate_dim / llama_1b_globals::matvec_block_size,
                          &Globals::down_weights,
                          &Globals::silu_out, /// TODO: CHECK
                          &Globals::hidden_states,
                          OPCODE_DownProjResidual,
                          OPCODE_DownProjResidual - 1,
                          Config>
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
                        Config>
    {
    };
}
