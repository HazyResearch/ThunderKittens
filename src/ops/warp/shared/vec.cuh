#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

/**
 * @file vec.cuh
 * @brief Defines unary operations on shared memory vectors.
 *
 * This file provides templated functions for performing unary operations on vectors
 * in shared memory, such as setting all elements to specific values like zero or infinity.
 * These operations are optimized for warp-level parallelism in CUDA kernels.
 */

namespace kittens {

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op The unary operation to apply.
 * @tparam T The type of the shared memory vector.
 * @param dst The shared memory vector to which the operation is applied.
 */
template<typename op, ducks::sv::all T>
__device__ static inline void st_unary_map(T &dst) {
    __syncwarp();
    #pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=WARP_SIZE) {
        auto col       = cur % T::length;
        auto row       = cur / T::length;
        auto idx       = row*T::length + col;
        dst[idx] = op::template op<typename T::dtype>(dst[idx]);
    }
}

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T The type of the shared memory vector.
 * @param dst The shared memory vector to be zeroed.
 */
template<ducks::sv::all T>
__device__ static inline void zero(T &dst)      { st_unary_map<base_ops::zero, T>(dst);      }

/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T The type of the shared memory vector.
 * @param dst The shared memory vector to be set to one.
 */
template<ducks::sv::all T>
__device__ static inline void one(T &dst)       { st_unary_map<base_ops::one, T>(dst);       }

/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T The type of the shared memory vector.
 * @param dst The shared memory vector to be set to positive infinity.
 */
template<ducks::sv::all T>
__device__ static inline void pos_infty(T &dst) { st_unary_map<base_ops::pos_infty, T>(dst); }

/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T The type of the shared memory vector.
 * @param dst The shared memory vector to be set to negative infinity.
 */
template<ducks::sv::all T>
__device__ static inline void neg_infty(T &dst) { st_unary_map<base_ops::neg_infty, T>(dst); }

}
