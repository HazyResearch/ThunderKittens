/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 */
template<ducks::sv::all SV>
__device__ static inline void load(SV &dst, const typename SV::dtype *src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = ::kittens::laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer];
    }
}
/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 */
template<ducks::sv::all SV>
__device__ static inline void store(typename SV::dtype *dst, const SV &src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = ::kittens::laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < src.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer]; // lmao it's identical
    }
}

}