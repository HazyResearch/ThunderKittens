#pragma once

#include "llama.cuh"

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
                int ret_order[] = {5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const globals &g, state<Config> &s)
            {
                for (int i = 0; i < 4; i++)
                {
                    kittens::init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
                }
                kittens::init_semaphore(outputs_arrived(s), 16); // outputs arrived.
                kittens::init_semaphore(activations_arrived(s), 1);
                s.record(1);
                return 6;
            }
        };
        struct loader
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((int *)s.scratch())[kittens::laneid()] = 0;
                kittens::warp::sync(); // done, now we can proceed to other things.
                if (kittens::laneid() < 4)
                {
                    s.wait_page_ready(get_weight_page(s, kittens::laneid()));
                    s.record(16 + kittens::laneid());
                    auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(s.pages[get_weight_page(s, kittens::laneid())]);
                    kittens::tma::expect(inputs_arrived(s, kittens::laneid()), weight_chunk);

                    auto &weights_global = g.*WeightsPtr; // object in global memory
                    kittens::tma::load_async(weight_chunk, weights_global, coord<>{inst.layer, inst.start_output_col, inst.start_reduction_col + 512 * laneid()}, inputs_arrived(s, laneid()));

                    // auto& weights_global = g.*WeightsPtr;      // object in global memory
                    // kittens::tma::load_async(weight_chunk, weights_global, coord<>{inst.layer, inst.start_output_col, inst.start_reduction_col + 512 * laneid()}, inputs_arrived(s, laneid()));
                }
                else if (kittens::laneid() == 31)
                {
                    int activation_page = get_activation_page(s);
                    s.wait_page_ready(activation_page);
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                        __nanosleep(20);
                    s.record(24);
                    auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[activation_page]);
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
                kittens::rt_bf<16, 128> weights, broadcast_activations;
                typename kittens::rt_bf<16, 128>::row_vec activations_vec;
                typename kittens::rt_bf<16, 128>::col_vec output_col_format;
                kittens::rv_bf<16> output;
                int group_id = kittens::warpgroup::groupid();
                int warp_id = kittens::warpgroup::warpid(); // id within the warpgroup
                wait(inputs_arrived(s, group_id), 0);
                if (laneid() == 0)
                    s.record(32 + warpid());
                // Reinterpret the page as a st_bf<16, 128>[4], which turns out to be a valid recast of the layout.
                int weight_page = get_weight_page(s, group_id);
                st_bf<16, 128>(&weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[weight_page]);
                kittens::warp::load(weights, weights_smem[warp_id]);
                kittens::warp::sync();
                kittens::warp::arrive(s.page_finished[weight_page], Config::NUM_CONSUMER_WARPS / 4); // this is called by each warp in the warpgroup
                // Next we need to load the activations
                wait(activations_arrived(s), 0);
                if (laneid() == 0)
                    s.record(64 + warpid());
                // reinterpret the activations page as sv_bf<128>[16]
                int activation_page = get_activation_page(s);
                kittens::sv_bf<128>(&activations_smem)[16] = reinterpret_cast<kittens::sv_bf<128>(&)[16]>(s.pages[activation_page]);
                kittens::warp::load(activations_vec, activations_smem[kittens::warpid()]);
                kittens::warp::sync();
                kittens::warp::arrive(s.page_finished[activation_page]); // just 1 is sufficient
                // broadcast this into a tile
                kittens::warp::broadcast_col(broadcast_activations, activations_vec);
                kittens::warp::mul(broadcast_activations, broadcast_activations, weights);
                kittens::warp::row_sum(output_col_format, broadcast_activations);
                kittens::warp::copy(output, output_col_format);
                // Now the first 16 threads have the output.
                if (laneid() < 16)
                { // this might be a bad idea but yolo, it's probably an okay start
                    // and fortunately this is code where ncu will tell us if it's bad..
                    atomicAdd(&((bf16 *)s.scratch())[kittens::laneid()], output[0][0]);
                }
                kittens::warp::sync();
                kittens::warp::arrive(outputs_arrived(s));
                if (kittens::group<16>::laneid() == 0)
                    s.record(124);
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                if (laneid() == 0)
                {
                    wait(outputs_arrived(s), 0);
                    s.record(125);
                    void *scratch = s.scratch();
                    sv_bf<16> &output = *reinterpret_cast<sv_bf<16> *>(scratch);

                    auto &OutputActivations = g.*OutputActivationsPtr; // object in global memory
                    kittens::tma::store_add_async(OutputActivations, output, {inst.output_block_idx});
                    kittens::tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                    s.record(126);
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
                if (laneid() == 0)
                    s.record(127);
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
