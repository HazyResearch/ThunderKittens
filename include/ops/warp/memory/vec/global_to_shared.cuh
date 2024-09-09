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
 * @param[in] idx The index of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void load(SV &dst, const GL &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (dst.length + WARP_THREADS*elem_per_transfer - 1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<SV>(idx);
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            move<float4>::ldg(tmp, &src_ptr[i*elem_per_transfer]);
            move<float4>::sts(&dst[i*elem_per_transfer], tmp);
        }
    }
}
/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The index of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void store(GL &dst, const SV &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (src.length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst.template get<SV>(idx);
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            move<float4>::lds(tmp, &src[i*elem_per_transfer]);
            move<float4>::stg(&dst_ptr[i*elem_per_transfer], tmp);
        }
    }
}

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam SV The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The index of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void load_async(SV &dst, const GL &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (dst.length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<SV>(idx);
    __syncwarp();
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                :
                : "l"((uint64_t)&dst[i*elem_per_transfer]), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

}