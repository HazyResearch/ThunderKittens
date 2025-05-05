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
                tma::load_async(weight_chunk, g.*WeightsPtr, coord<>{inst.layer, (inst.start_block_idx + iter) * Globals::matvec_block_size, inst.start_reduction_col + 512 * col_idx}, sem);
            }

            static __device__ inline void store(state<Config> &s, const globals &g, parsed_instruction &inst, int output_idx, int output_stage, semaphore &sem, int bit)
            {
                wait(sem, bit);

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
                    tma::store_add_async<cache_policy::NORMAL>(OutputActivations, output_smem_bf, {block_idx});
                    tma::store_async_wait();
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
                    parsed_instruction inst{s};

                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);

                    int activation_page = pipeline::get_activation_page(s);
                    s.wait_page_ready(activation_page);

                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                    {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                    s.record(TEVENT_DONE_GMEM_WAIT);

                    auto &activations = pipeline::get_activations(s);
                    tma::expect(pipeline::activations_arrived(s), activations);

                    auto &InputActivations = g.*InputActivationsPtr; // object in global memory
                    tma::load_async(activations, InputActivations, coord<>{inst.start_reduction_col}, pipeline::activations_arrived(s));
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                wait(pipeline::activations_arrived(s), 0);

                using sv_t = sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
                using rv_t = rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;

                sv_t(&activations_smem)[Config::NUM_CONSUMER_WARPS] = reinterpret_cast<sv_t(&)[Config::NUM_CONSUMER_WARPS]>(pipeline::get_activations(s));

                rv_t activations_vec;
                warp::load(activations_vec, activations_smem[warpid()]);
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
                    parsed_instruction inst{s};

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!

                    asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);
                }
            }
        };
    };

    // template <
    //     int _EXPECTED_ARRIVAL_COUNT,
    //     auto WeightsPtr,
    //     auto InputActivationsPtr,
    //     auto OutputActivationsPtr,
    //     int _opcode,
    //     int _prev_opcode = 0,
    //     typename Config = kittens::prototype::vm::default_config>

    // struct MatVecAddOp
    // {
    //     static constexpr int opcode = _opcode;
    //     static constexpr int prev_opcode = _prev_opcode;
    //     static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;

    //     static constexpr int NUM_WEIGHT_PAGES = 4;
    //     static constexpr int PAGE_WEIGHT_START = 0;
    //     static constexpr int PAGE_ACTIVATION = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;
    //     static constexpr int PAGE_COUNT = PAGE_ACTIVATION + 1; // 5

    //     static constexpr int REDUCTION_DIM_PER_WARP = llama_1b_globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    //     struct parsed_instruction
    //     {
    //         int layer, output_block_idx, reduction_block_idx, start_output_col, start_reduction_col;
    //         __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
    //         {
    //             layer = instruction[1];               // in units of 1
    //             output_block_idx = instruction[2];    // in units of 1 (0, 16, 32, ..., 2032)
    //             reduction_block_idx = instruction[3]; // in units of 2048 (0, 2048, 4096, 6144)
    //             start_output_col = output_block_idx * llama_1b_globals::matvec_block_size;
    //             start_reduction_col = reduction_block_idx * llama_1b_globals::hidden_dim;
    //         }
    //         __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    //     };
    //     static __device__ inline parsed_instruction parse_instruction(const globals &g, state<Config> &s)
    //     {
    //         return parsed_instruction{s.instruction()[1], s.instruction()[2]};
    //     }
    //     __device__ static inline kittens::semaphore &inputs_arrived(state<Config> &s, int id)
    //     {
    //         return s.semaphores()[id];
    //     }
    //     __device__ static inline kittens::semaphore &outputs_arrived(state<Config> &s)
    //     {
    //         return s.semaphores()[4];
    //     }
    //     __device__ static inline kittens::semaphore &activations_arrived(state<Config> &s)
    //     {
    //         return s.semaphores()[5];
    //     }
    //     __device__ static inline int get_weight_page(state<Config> &s, int offset)
    //     {
    //         return s.pid(offset + 1);
    //     }
    //     __device__ static inline int get_activation_page(state<Config> &s)
    //     {
    //         return s.pid(0);
    //     }

    //     struct controller
    //     {
    //         static __device__ int release_lid(const globals &g, typename Config::instruction_t &instruction, int &query)
    //         {
    //             int ret_order[] = {5, 6, 7, 8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3};
    //             return ret_order[query];
    //         }
    //         static __device__ int init_semaphores(const globals &g, state<Config> &s)
    //         {
    //             for (int i = 0; i < 4; i++)
    //             {
    //                 kittens::init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
    //             }
    //             kittens::init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS); // outputs arrived.
    //             kittens::init_semaphore(activations_arrived(s), 1);
    //             return 6;
    //         }
    //     };
    //     struct loader
    //     {
    //         static __device__ void run(const globals &g, state<Config> &s)
    //         {
    //             parsed_instruction inst{s};
    //             // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
    //             ((uint64_t *)s.scratch())[kittens::laneid()] = 0;
    //             kittens::warp::sync(); // done, now we can proceed to other things.

    //             if (kittens::laneid() == 0)
    //             {
    //                 for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
    //                 {
    //                     s.wait_page_ready(get_weight_page(s, i));
    //                     auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(s.pages[get_weight_page(s, i)]);
    //                     s.record(TEVENT_TRIPLES_START + i);
    //                     kittens::tma::expect(inputs_arrived(s, i), weight_chunk);

    //                     auto &weights_global = g.*WeightsPtr; // object in global memory
    //                     kittens::tma::load_async(weight_chunk, weights_global, coord<>{inst.layer, inst.start_output_col, inst.start_reduction_col + 512 * i}, inputs_arrived(s, i));
    //                 }
    //             }
    //             else if (kittens::laneid() >= PAGE_COUNT && kittens::laneid() < Config::NUM_PAGES)
    //             {
    //                 int unused_page = s.pid(kittens::laneid());
    //                 s.wait_page_ready(unused_page);
    //                 s.finish_page(unused_page, Config::NUM_CONSUMER_WARPS);
    //             }
    //         }
    //     };
    //     struct launcher
    //     { // launches mma's
    //         // launcher does nothing here, since this doesn't use tensor cores.
    //         static __device__ void run(const globals &g, state<Config> &s)
    //         {
    //             if (laneid() == 0)
    //             {
    //                 parsed_instruction inst{s};

    //                 s.wait_tensor_ready();
    //                 arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);

    //                 int activation_page = get_activation_page(s);
    //                 s.wait_page_ready(activation_page);

    //                 s.record(TEVENT_AT_GMEM_WAIT);
    //                 while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
    //                 {
    //                     __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
    //                 }
    //                 s.record(TEVENT_DONE_GMEM_WAIT);

    //                 auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[activation_page]);
    //                 s.record(TEVENT_TRIPLES_START + 4);
    //                 kittens::tma::expect(activations_arrived(s), activations);

    //                 auto &InputActivations = g.*InputActivationsPtr; // object in global memory
    //                 kittens::tma::load_async(activations, InputActivations, coord<>{inst.start_reduction_col}, activations_arrived(s));
    //             }
    //         }
    //     };
    //     struct consumer
    //     {
    //         static __device__ void run(const globals &g, state<Config> &s)
    //         {
    //             using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
    //             using float_rv_t = rv_fl<16>;

    //             typename float_rt_t::row_vec activations_vec;

    //             static_assert(Config::NUM_CONSUMER_WARPS % NUM_WEIGHT_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by NUM_WEIGHT_PAGES");
    //             constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / NUM_WEIGHT_PAGES;

    //             int page_index = warpid() / WARPS_PER_PAGE;

    //             using sv_slice_t = sv_bf<REDUCTION_DIM_PER_WARP>;

    //             wait(activations_arrived(s), 0);

    //             int activation_page = get_activation_page(s);
    //             sv_slice_t(&activations_smem)[Config::NUM_CONSUMER_WARPS] = reinterpret_cast<sv_slice_t(&)[Config::NUM_CONSUMER_WARPS]>(s.pages[activation_page]);

    //             warp::load(activations_vec, activations_smem[warpid()]);
    //             warp::sync();

    //             s.warp_finish_page(activation_page, 1);

    //             matvec<float_rt_t, WARPS_PER_PAGE>(g, s, activations_vec, inputs_arrived(s, page_index), get_weight_page(s, page_index), 0);

    //             warp::sync();
    //             warp::arrive(outputs_arrived(s));

    //             for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
    //             {
    //                 s.warp_finish_page(get_weight_page(s, i), 1);
    //             }
    //         }
    //     };
    //     struct storer
    //     {
    //         // Uses 4 full pages for outputs.
    //         static __device__ void run(const globals &g, state<Config> &s)
    //         {
    //             if (kittens::laneid() == 0)
    //             {
    //                 s.record(TEVENT_TRIPLES_STORE_START);
    //             }

    //             parsed_instruction inst{s};

    //             void *scratch = s.scratch();

    //             // Convert to bf and put back in shared memory
    //             sv_fl<16> &output = *reinterpret_cast<sv_fl<16> *>(scratch);
    //             sv_bf<16> &output_bf = *reinterpret_cast<sv_bf<16> *>(scratch);

    //             rv_bf<16> output_reg_bf;

    //             wait(outputs_arrived(s), 0);
    //             if (kittens::laneid() == 0)
    //             {
    //                 s.record(TEVENT_TRIPLES_OUTPUT_READY);
    //             }

    //             warp::load(output_reg_bf, output);
    //             warp::sync();
    //             warp::store(output_bf, output_reg_bf);
    //             warp::sync();

    //             if (laneid() == 0)
    //             {
    //                 auto &OutputActivations = g.*OutputActivationsPtr; // object in global memory
    //                 kittens::tma::store_add_async(OutputActivations, output_bf, {inst.output_block_idx});
    //                 kittens::tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
    //             }
    //             kittens::warp::sync();
    //             asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
    //             if (laneid() == 0)
    //             {
    //                 atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], 1);
    //             }
    //         }
    //     };
    // };

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
