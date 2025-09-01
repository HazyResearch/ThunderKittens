#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template <typename config, typename globals>
__device__ void inline store_timings(int *timings, int instruction_index, const globals &g) {
    constexpr int bytes = config::TIMING_WIDTH * sizeof(int);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(timings));

    if constexpr (ducks::gl::all<decltype(g.instructions)>)  {
        uint64_t dst_ptr = (uint64_t)(&g.timings[kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}]);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
            :
            : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
            : "memory");
    } else if constexpr (ducks::pgl::all<decltype(g.instructions)>) {
        uint64_t dst_ptr = (uint64_t)(&g.timings[g.dev_idx][kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}]);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
            :
            : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
            : "memory");
    }
    kittens::tma::store_commit_group();
}

template <typename config, typename globals>
__device__ void inline store_timings_and_reset(int *timings, int instruction_index, const globals &g) {
    if(laneid() == 0) {
        store_timings<config, globals>(timings, instruction_index, g);
        kittens::tma::store_async_read_wait();
#ifdef KITTENS_BLACKWELL
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(timings));
        asm volatile("st.bulk.weak [%0], %1, 0;\n" ::"r"(src_ptr), "n"(config::TIMING_WIDTH * sizeof(int))); // Reinitialize timing memory as zeros.   
#endif
    }
#ifndef KITTENS_BLACKWELL
    __syncwarp();
    for(int i = laneid(); i < config::TIMING_WIDTH; i += WARP_THREADS) {
        timings[i] = 0;
    }
#endif
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
