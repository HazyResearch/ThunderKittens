/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load(SV &dst, const GL &src, const COORD &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (dst.length + WARP_THREADS*elem_per_transfer - 1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[i*elem_per_transfer]);
            move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
        }
    }
}

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store(const GL &dst, const SV &src, const COORD &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (src.length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            move<float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer);
            move<float4>::stg((float4*)&dst_ptr[i*elem_per_transfer], tmp);
        }
    }
}

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam SV The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>, bool should_commit_group=true>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = (dst.length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    __syncwarp();
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                :: "r"(dst_ptr + uint32_t(sizeof(typename SV::dtype))*i*elem_per_transfer), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    if constexpr (should_commit_group) {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
}

/**
 * @brief Asynchronously loads a vector from global to shared memory. Additionally, each thread will
 * signal the semaphore as it completes its load. For example, if you invoke load_async from a warp
 * with a semaphore, when all threads are completed, the sempahore's completed arrivals count
 * will be 32.
 * 
 * Example Usage;
 *     semaphore bar;
 *     init_semaphore(bar, 32, 0); // 32 threads in a warp
 *     load_async(dst, src, idx, bar);
 *     wait(bar, 0); // ding! memory arrived from all threads in the warp
 *
 * @tparam ST The type of the shared vector.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coordinate of the vector in the global memory array.
 * @param[in] bar The semaphore to signal.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore &bar) {
    load_async<SV, GL, COORD, false>(dst, src, idx);
    asm volatile(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" 
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar)))
        : "memory"
    );
}


}