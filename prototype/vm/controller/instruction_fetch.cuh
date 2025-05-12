#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens
{
    namespace prototype
    {
        namespace vm
        {
            namespace controller
            {
                template <typename config, typename globals>
                __device__ void inline load_instructions(int *instruction, int instruction_index, const globals &g)
                {
                    auto laneid = ::kittens::laneid();
                    // if (laneid > 0)
                    // {
                    //     return;
                    // }

                    auto src_ptr = &g.instructions[kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
                    // static assert it's an int*
                    static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");

                    static_assert(config::INSTRUCTION_WIDTH <= 32);

                    if (laneid < config::INSTRUCTION_WIDTH)
                    {
                        instruction[laneid] = src_ptr[laneid];
                    }

                    // // Load instructions using a simple for loop from global memory
                    // for (int i = 0; i < config::INSTRUCTION_WIDTH; i++)
                    // {
                    //     instruction[i] = src_ptr[i];
                    // }

                    // arrive(bar, 1);

                    // constexpr int bytes = config::INSTRUCTION_WIDTH*sizeof(int);
                    // ::kittens::tma::expect_bytes(bar, bytes);
                    // uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
                    // uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(instruction));
                    // uint64_t src_ptr  = (uint64_t)(&g.instructions[kittens::coord<>{(int)(blockIdx.x), instruction_index, 0}]);
                    // asm volatile (
                    //     "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];\n"
                    //     :
                    //     : "r"(dst_ptr), "l"(src_ptr), "r"(mbar_ptr), "n"(bytes)
                    //     : "memory"
                    // );
                }

                template <typename config, typename globals>
                __device__ void inline instruction_fetch_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms)
                {
                    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
                    int num_iters = g.instructions.rows();
                    for (kvms.instruction_index = 0, kvms.instruction_ring = 0; kvms.instruction_index < num_iters; kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring))
                    {
                        int phasebit = (kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES - 1) & 1;
                        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES)
                            wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);
                        load_instructions<config, globals>(&kvms.instruction()[0], kvms.instruction_index, g, kvms.instruction_arrived[kvms.instruction_ring]);

                        // TODO: fix after fixing timings arrival
                        // if(kvms.instruction_index < config::INSTRUCTION_PIPELINE_STAGES) {
                        //     /* This requires some explanation.
                        //      *
                        //      * For an instruction stage to be ready to use, two things must happen:
                        //      * 1. The instruction must be loaded.
                        //      * 2. Any previous timing writeout must be finished.
                        //      *
                        //      * Correspondingly, the instruction_arrived semaphore takes two separate arrivals.
                        //      * One is triggered by this thread, and the other is triggered by the timing store thread.
                        //      * However, for the first N stages, there is no timing store, so we need to trigger the arrival manually.
                        //      */
                        //     arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
                        // }
                    }
                }

            } // namespace controller
        } // namespace vm
    } // namespace prototype
} // namespace kittens
