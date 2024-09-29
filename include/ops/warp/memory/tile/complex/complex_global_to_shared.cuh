/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

#include "../global_to_shared.cuh"

namespace kittens {

/**
 * @brief Loads data from global memory into a complex shared memory tile with a row layout.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] dst The destination complex shared memory tile.
 * @param[in] resrc The source global memory array for the real component.
 * @param[in] imsrc The source global memory array for the imaginary component.
 * @param re_row_stride[in] The stride between rows in the source real component array.
 * @param im_row_stride[in] The stride between rows in the source imaginary component array.
 */
template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void load(CST &dst, const CGL &src, const coord &idx) {
    load(dst.real, src.real, idx);
    load(dst.imag, src.imag, idx);
}

/**
 * @brief Stores bf16 data from a complex shared memory tile with a row layout into global memory.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] redst The destination global memory array for the real component.
 * @param[out] imdst The destination global memory array for the imaginary component.
 * @param[in] src The source complex shared memory tile.
 * @param re_row_stride[in] The stride between rows in the destination real component array.
 * @param im_row_stride[in] The stride between rows in the destination imaginary component array.
 */
template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void store(const CGL &dst, CST &src, const coord &idx) {
    store(dst.real, src.real, idx);
    store(dst.imag, src.imag, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a complex shared memory tile with a row layout using CUDA barriers.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] resrc The source global memory array for the real component.
 * @param[in] imsrc The source global memory array for the imaginary component.
 * @param re_row_stride[in] The stride between rows in the real component source array.
 * @param im_row_stride[in] The stride between rows in the imaginary component source array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void load_async(CST &dst, const CGL &src, const coord &idx) {
    load_async(dst.real, src.real, idx);
    load_async(dst.imag, src.imag, idx);
}
}