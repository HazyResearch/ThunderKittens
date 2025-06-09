/**
 * @file
 * @brief Conversions between shared tile types.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"

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
template<typename T, typename U, int _height, int _width>
static inline void copy(st<T, _height, _width> &dst, const st<U, _height, _width> &src) {
    #pragma unroll
    for(int i = laneid(); i < dst.num_elements; i+=kittens::WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
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
* @param row_idx The row coord of the subtile, in units of subtile_height*16 elements.
* @param col_idx The col coord of the subtile, in units of subtile_width*16 elements.
* @return A reference to the subtile.
*
* @note The subtile {height, width} must evenly divide the tile {height, width}.
*/
template <int subtile_rows, int subtile_cols, ducks::st::all ST>
inline st_subtile<ST, subtile_rows, subtile_cols>
subtile_inplace(ST &src, sycl::int2 rowcol) {
    using T = typename ST::dtype;
    static_assert(subtile_rows % TILE_ROW_DIM<T> == 0);
    static_assert(subtile_cols % TILE_COL_DIM<T> == 0);
    static_assert(ST::height % (subtile_rows/TILE_ROW_DIM<T>) == 0);
    static_assert(ST::width % (subtile_cols/TILE_COL_DIM<T>) == 0);
    static_assert(ST::height == ST::underlying_height && ST::width == ST::underlying_width); // must be a real ST, no recursive subtiles.
    return st_subtile<ST, subtile_rows, subtile_cols>(src, rowcol);
}

} // namespace kittens