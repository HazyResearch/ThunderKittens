/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

#include "../global_to_register.cuh"

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
 template<ducks::rt::row_layout RT, typename U>
__device__ inline static void load(RT &dst, const U *src, const int row_stride) {
template<ducks::st::complex CST>
__device__ static inline void load(CST &dst, const typename CST::dtype::dtype *resrc, const typename CST::dtype::dtype *imsrc, const int re_row_stride, const int im_row_stride) {
    warpgroup::load(dst.real, resrc, re_row_stride);
    warpgroup::load(dst.imag, imsrc, im_row_stride);
}