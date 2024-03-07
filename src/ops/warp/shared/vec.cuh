
#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"


namespace kittens {


/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst Destination vector to apply the unary operation to.
 */
template<typename op, st_vec_type T>
__device__ 
__device__ static inline void unary_map(T &dst) {
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
 * @tparam T Shared memory vector type.
 * @param dst Destination vector to be set to zero.
 */
template<st_vec_type T>
__device__ static inline void zero(T &dst)      { unary_map<base_ops::zero, T>(dst);      }
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst Destination vector to be set to one.
 */
template<st_vec_type T>
__device__ static inline void one(T &dst)       { unary_map<base_ops::one, T>(dst);       }
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst Destination vector to be set to positive infinity.
 */
template<st_vec_type T>
__device__ static inline void pos_infty(T &dst) { unary_map<base_ops::pos_infty, T>(dst); }
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst Destination vector to be set to negative infinity.
 */
template<st_vec_type T>
__device__ static inline void neg_infty(T &dst) { unary_map<base_ops::neg_infty, T>(dst); }


}