#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals> __device__ void inline store_timings(int *timings, int instruction_index, const globals &g) {
    constexpr int bytes = config::TIMING_WIDTH*sizeof(int);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(timings));
    uint64_t dst_ptr  = (uint64_t)(&g.timings[kittens::coord<>{(int)(blockIdx.x), instruction_index, 0}]);
    asm volatile (
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :
        : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
        : "memory"
    );
    kittens::tma::store_commit_group();
}

template<typename config, typename globals> __device__ void inline timings_store_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
    int num_iters = g.instructions.rows();
    uint32_t semaphore_bitfield = 0xFFFF0000;
    for(kvms.instruction_index = 0, kvms.instruction_ring = 0;
        kvms.instruction_index < num_iters;
        kvms.instruction_index++
    ) {
        wait(kvms.instruction_finished[kvms.instruction_ring], get_phasebit<0>(semaphore_bitfield, kvms.instruction_ring));
        update_phasebit<0>(semaphore_bitfield, kvms.instruction_ring);
        store_timings<config, globals>(&kvms.timing()[0], kvms.instruction_index, g);
        kittens::tma::store_async_read_wait();
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&kvms.timing()[0]));
        asm volatile("st.bulk.weak [%0], %1, 0;\n" :: "r"(src_ptr), "n"(config::TIMING_WIDTH*sizeof(int))); // Reinitialize timing memory as zeros.
        kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring); // advance before arrival
        arrive(kvms.instruction_arrived[kvms.instruction_ring]);
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
