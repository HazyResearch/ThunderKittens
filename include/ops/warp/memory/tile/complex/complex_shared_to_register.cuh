/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

#include "../shared_to_register.cuh"

namespace kittens {

/**
 * @brief Load data from a complex shared tile into a complex register tile.
 *
 * @tparam CRT The complex register tile type
 * @tparam CST The complex shared tile type
 * @param dst[out] The destination complex register tile.
 * @param src[in]  The source complex shared tile.
 */
template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void load(CRT &dst, const CST &src) {
    load(dst.real, src.real);
    load(dst.imag, src.imag);
}

/**
 * @brief Store data into a complex shared tile from a complex register tile.
 *
 * @tparam RT The complex register tile type
 * @tparam ST The complex shared tile type
 * @param dst[out] The destination complex shared tile.
 * @param src[in]  The source complex register tile.
 */
template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void store(CST &dst, const CRT &src) {
    store(dst.real, src.real);
    store(dst.imag, src.imag);
}



}