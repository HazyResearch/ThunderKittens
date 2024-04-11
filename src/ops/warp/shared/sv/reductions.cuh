#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Performs a reduction operation on elements of a shared memory vector within a warp.
 *
 * This function applies a specified operation to reduce the elements of a shared memory vector `src` to a single value.
 * The result is stored in `accum`. If the `reset` parameter is true, the reduction includes an initial value `src_accum`.
 * The reduction operation is performed in a warp-wide context, ensuring synchronization between threads in the warp.
 *
 * @tparam op The operation to perform on the elements. Must provide a static `op` method.
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @tparam reset A boolean flag indicating whether to include an initial value in the reduction.
 * @param[out] accum The result of the reduction operation.
 * @param[in] src The shared memory vector to reduce.
 * @param[in] src_accum The initial value to include in the reduction if `reset` is false.
 */
template<typename op, ducks::sv::all SV, bool reset>
__device__ static inline void reduce(typename SV::dtype &accum, const SV &src, const typename SV::dtype &src_accum) {
    using T = SV::dtype;
    int laneid = kittens::laneid();
    accum = laneid < src.length ? src[laneid] : 0; // initialize a register accumulator
    __syncwarp();
    #pragma unroll
    for(int i = laneid + kittens::WARP_THREADS; i < src.length; i+=kittens::WARP_THREADS) {
        accum = op::template op<T>(accum, src[i]);
        __syncwarp();
    }
    // We can now reduce within the warp.
    if constexpr (src.length > 16) {
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
        __syncwarp();
    }
    accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
    __syncwarp();
    accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
    __syncwarp();
    accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
    __syncwarp();
    accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
    __syncwarp();
    if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
    // broadcast to all threads in the warp.
    accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0); // everyone takes from warp leader
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Finds the maximum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The shared memory vector to find the maximum in.
 */
template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src) {
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
}

/**
 * @brief Finds the minimum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The shared memory vector to find the minimum in.
 */
template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src) {
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The shared memory vector to sum.
 */
template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src) {
    reduce<base_ops::add, SV, true>(sum_val, src, sum_val);
}

/**
 * @brief Calculates the product of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The shared memory vector to multiply.
 */
template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src) {
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
}

/**
 * @brief Finds the minimum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::add, SV, false>(sum_val, src, src_accum);
}

/**
 * @brief Calculates the product of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
}
}