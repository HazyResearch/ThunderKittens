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
 template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    // Internally will use the correct load() method for row and column types
    load<axis, CRT, CGL, COORD>(dst.real, src.real, idx);
    load<axis, CRT, CGL, COORD>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL>
__device__ inline static void load(CRT &dst, const CGL &src, const coord<CRT> &idx) {
    load<2, CRT, CGL, coord<CRT>>(dst, src, idx);
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
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    // Internally will use the correct load() method for row and column types
    store<axis, CGL, CRT, COORD>(dst.real, src.real, idx);
    store<axis, CGL, CRT, COORD>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL>
__device__ inline static void store(CGL &dst, const CRT &src, const coord<CRT> &idx) {
    store<2, CGL, CRT, coord<CRT>>(dst, src, idx);
}
}