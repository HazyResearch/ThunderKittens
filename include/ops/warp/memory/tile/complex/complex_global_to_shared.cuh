/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

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
template<ducks::st::complex CST>
__device__ static inline void load(CST &dst, const typename CST::dtype::dtype *resrc, const typename CST::dtype::dtype *imsrc, const int re_row_stride, const int im_row_stride) {
    load(dst.real, resrc, re_row_stride);
    load(dst.imag, imsrc, im_row_stride);
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
template<ducks::st::complex CST>
__device__ static inline void store(const typename CST::dtype::dtype *redst, const typename CST::dtype::dtype *imdst, CST &src, const int re_row_stride, const int im_row_stride) {
    store(redst, src.real, re_row_stride);
    store(imdst, src.imag, im_row_stride);
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
template<ducks::st::complex CST>
__device__ static inline void load_async(CST &dst, const typename CST::dtype::dtype *resrc, const typename CST::dtype::dtype *imsrc, const int re_row_stride, const int im_row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    load_async(dst.real, resrc, re_row_stride, barrier);
    load_async(dst.imag, imsrc, im_row_stride, barrier);
}

/**
 * @brief Asynchronously stores data from a complex shared memory tile with a row layout into global memory using CUDA barriers.
 *
 * @tparam ST The type of the shared tile
 * @param[out] redst The destination real component global memory array.
 * @param[out] imdst The destination imaginary component global memory array.
 * @param[in] src The source shared memory tile.
 * @param re_row_stride[in] The stride between rows in the real component destination array.
 * @param im_row_stride[in] The stride between rows in the imaginary component destination array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::st::complex CST>
__device__ static inline void store_async(typename CST::dtype::dtype *redst, typename CST::dtype::dtype *imdst, const CST &src, const int re_row_stride, const int im_row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    store_async(redst, src.real, re_row_stride, barrier);
    store_async(imdst, src.imag, im_row_stride, barrier);
}

}