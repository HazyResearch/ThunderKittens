#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {
template<typename config, typename globals>
__device__ void inline load_instructions(int *instruction, int task_iter, const globals &g, kittens::semaphore &bar) {
    constexpr int bytes = config::INSTRUCTION_WIDTH*sizeof(int);
    tma::expect_bytes(bar, bytes);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(instruction));
    uint64_t src_ptr  = (uint64_t)(&g.instructions[kittens::coord<>{(int)(blockIdx.x), task_iter, 0}]);

    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mbar_ptr), "r"(bytes)); // Expect
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];\n"
        :
        : "r"(dst_ptr), "l"(src_ptr), "r"(mbar_ptr), "n"(bytes)
        : "memory"
    );
}

template<typename config, typename globals> __device__ void inline instruction_fetch_loop(const globals &g, state<config> &cs) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
    int num_iters = g.instructions.rows();
    uint32_t semaphore_bitfield = 0xFFFF0000;
    for(cs.task_iter = 0, cs.ring = 0; cs.task_iter < num_iters; cs.task_iter++, cs.ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(cs.ring)) {
        wait(cs.instruction_finished[cs.ring], get_phasebit<1>(semaphore_bitfield, cs.ring));
        update_phasebit<1>(semaphore_bitfield, cs.ring);
        load_instructions<config, globals>(&cs.instruction[cs.ring][0], cs.task_iter, g, cs.instruction_arrived[cs.ring]);
        if(cs.task_iter < config::INSTRUCTION_PIPELINE_STAGES) {
            /* This requires some explanation.
             * 
             * For an instruction stage to be ready to use, two things must happen:
             * 1. The instruction must be loaded.
             * 2. Any previous timing writeout must be finished.
             * 
             * Correspondingly, the instruction_arrived semaphore takes two separate arrivals.
             * One is triggered by this thread, and the other is triggered by the timing store thread.
             * However, for the first N stages, there is no timing store, so we need to trigger the arrival manually.
             */
            arrive(cs.instruction_arrived[cs.ring], 1);
        }
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
