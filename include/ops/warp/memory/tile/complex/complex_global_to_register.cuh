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
 * @brief Load data from source arrays into a complex-type tile.
 *
 * @tparam CRT The complex tile type.
 * @tparam U The data type of the source arrays.
 * @param dst[out] The destination tile to load data into.
 * @param resrc[in] The source array to load the real component data from.
 * @param imsrc[in] The source array to load the imaginary component data from.
 * @param re_row_stride[in] The stride in elements between rows in the real component source array.
 * @param im_row_stride[in] The stride in elements between rows in the imaginary component source array.
 */
 template<ducks::rt::complex CRT, typename U>
__device__ inline static void load(CRT &dst, const U *resrc, const U *imsrc, const int re_row_stride, const int im_row_stride) {
    // Internally will use the correct load() method for row and column types
    load(dst.real, resrc, re_row_stride);
    load(dst.imag, imsrc, im_row_stride);
}

/**
 * @brief Store data from a complex register tile to destination arrays in global memory.
 *
 * @tparam CRT The complex tile type.
 * @tparam U The data type of the destination arrays.
 * @param redst[out] The destination array in global memory to store the real component data into.
 * @param imdst[out] The destination array in global memory to store the imaginary component data into.
 * @param src[in] The source register tile to store data from.
 * @param re_row_stride[in] The stride in elements between rows in the real component destination array.
 * @param im_row_stride[in] The stride in elements between rows in the imaginary component destination array.
 */
template<ducks::rt::complex CRT, typename U>
__device__ inline static void store(U *redst, U *imdst, const CRT &src, const int re_row_stride, const int im_row_stride) {
    // Internally will use the correct load() method for row and column types
    store(redst, src.real, re_row_stride);
    store(imdst, src.imag, im_row_stride);
}


}