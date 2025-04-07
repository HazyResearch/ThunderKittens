#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals> __device__ void inline store_timings(int *timings, int task_iter, const globals &g) {
    constexpr int bytes = config::TIMING_EVENTS*sizeof(int);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(timings));
    uint64_t dst_ptr  = (uint64_t)(&g.timings[kittens::coord<>{(int)(blockIdx.x), task_iter, 0}]);
    asm volatile (
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :
        : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
        : "memory"
    );
    kittens::tma::store_commit_group();
}

template<typename config, typename globals> __device__ void inline timings_store_loop(const globals &g, state<config> &cs) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
    int num_iters = g.instructions.rows();
    uint32_t semaphore_bitfield = 0xFFFF0000;
    for(cs.task_iter = 0, cs.ring = 0; cs.task_iter < num_iters; cs.task_iter++, cs.ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(cs.ring)) {
        wait(cs.instruction_finished[cs.ring], get_phasebit<0>(semaphore_bitfield, cs.ring));
        update_phasebit<0>(semaphore_bitfield, cs.ring);
        store_timings<config, globals>(&cs.timings[cs.ring][0], cs.task_iter, g);
        tma::store_async_read_wait();
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&cs.timings[cs.ring][0]));
        asm volatile("st.bulk.weak [%0], %1, 0;\n" :: "r"(src_ptr), "n"(config::TIMING_EVENTS*sizeof(int))); // Reinitialize timing memory as zeros.
        arrive(cs.instruction_arrived[cs.ring]);
        cs.ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(cs.ring);
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
