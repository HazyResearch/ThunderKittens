/**
 * @file
 * @brief Conversions between shared tile types.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  COPIES  ---------- */

/**
 * @brief Copies data from one shared memory tile to another, potentially with different data types and layouts.
 *
 * @tparam T The data type of the destination tile.
 * @tparam U The data type of the source tile.
 * @tparam _height The height of the tile.
 * @tparam _width The width of the tile.
 * @tparam L1 The layout of the destination tile.
 * @tparam L2 The layout of the source tile.
 * @param[out] dst The destination tile.
 * @param[in] src The source tile.
 */
template<typename T, typename U, int _height, int _width, ducks::st_layout::all L1, ducks::st_layout::all L2>
__device__ static inline void copy(st<T, _height, _width, L1> &dst, const st<U, _height, _width, L2> &src) {
    using T2 = base_types::packing<T>::packed_type;
    using U2 = base_types::packing<U>::packed_type;
    int lane = threadIdx.x % 32;
    if constexpr (std::is_same_v<L1, L2>) { // if same layout can just do a simple copy
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_THREADS) {
            dst[i] = base_types::convertor<T, U>::convert(src[i]);
        }
    }
    else { // otherwise we need to actually do indexing calculations :(
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_THREADS) {
            int row = i/dst.cols;
            int col = i%dst.cols;
            dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
        }
    }
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given shared tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam subtile_width The width of the subtile.
* @tparam ST The type of the input tile, which must satisfy the ducks::st::all concept.
* @param src The input tile.
* @param row_idx The row index of the subtile, in units of subtile_height*16 elements.
* @param col_idx The col index of the subtile, in units of subtile_width*16 elements.
* @return A reference to the subtile.
*
* @note The subtile {height, width} must evenly divide the tile {height, width}.
*/
template<int subtile_height, int subtile_width, ducks::st::all ST>
__device__ inline typename ST::subtile<subtile_height, subtile_width> subtile_inplace(ST &src, int row_idx, int col_idx) {
    static_assert(ST::height % subtile_height == 0);
    static_assert(ST::width % subtile_width == 0);
    return typename ST::subtile<subtile_height, subtile_width>(
        &src[0], subtile_height*16*row_idx, subtile_width*16*col_idx
    );
}

} // namespace kittens