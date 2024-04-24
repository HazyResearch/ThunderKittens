/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */

#pragma once

#include "../../common/common.cuh"
#include "st_layout.cuh"
#include "sv.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace st {
/**
 * @brief A dummy type used to identify shared tiles.
 * 
 * For a type to quack like an st, it should define its identifier as ducks::st::identifier.
 * If a type quacks like ducks::st::identifier, it will be treated as an st by compiler checks.
 * This is particularly useful for subtiles on challenging layouts.
 */
struct identifier {};
} // namespace st
} // namespace ducks

// Forward declaration of subtile
template<
    typename _T,
    int _underlying_height,
    int _underlying_width,
    ducks::st_layout::all _layout,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile;

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _height The height of the tile in units of 16-element subtiles.
 * @tparam _width The width of the tile in units of 16-element subtiles.
 * @tparam _layout The memory layout of the tile.
 */
template<typename _T, int _height, int _width, ducks::st_layout::all _layout>
struct st {
    using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
    using layout = _layout; ///< Memory layout of the tile.
    using dtype = _T; ///< Data type of the elements in the tile.

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_height        = _height;
    static constexpr int underlying_width         = _width;
    static constexpr int underlying_rows          = underlying_height * kittens::TILE_DIM;
    static constexpr int underlying_cols          = underlying_width  * kittens::TILE_DIM;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int height              = _height; ///< Height of the tile in terms of 16-element subtiles.
    static constexpr int width               = _width; ///< Width of the tile in terms of 16-element subtiles.
    static constexpr int rows                = height * kittens::TILE_DIM; ///< Total number of rows in the tile.
    static constexpr int cols                = width  * kittens::TILE_DIM; ///< Total number of cols in the tile.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    static_assert(
        !std::is_same_v<layout, ducks::st_layout::xor_swizzle> || width == 1 || width == 2 || width == 4 || width == 8 || width == 16 || width == 32,
        "For XOR swizzled modes, shared tile width must be a power of 2."
    ); // XOR swizzling only works with a few particular layout dimensions.

    // wgmma layout with swizzling
    dtype data[rows*cols]; ///< Raw data storage for the tile.

    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled or strangely ordered layouts.
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return data[detail::shared_indexer<height, width, layout>::idx(rowcol.x, rowcol.y)];
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return data[detail::shared_indexer<height, width, layout>::idx(rowcol.x, rowcol.y)];
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    // vector types
    using col_vec = sv<dtype, height>; ///< Column vector type for this tile
    using row_vec = sv<dtype, width>; ///< Row vector type for this tile
    template<int subtile_height, int subtile_width> using subtile = st_subtile<
        dtype, height, width, layout,
        subtile_height, subtile_width
    >; ///< A templated subtile type wrapper for this tile.
};


/**
 * @brief A reference into a chunk of shared tile memory.
 *
 * The st_subtile is a drop-in replacement for an st which internally
 * references the appropriate memory while performing minimal address
 * calculations. You should never create this directly, but instead
 * have subtile_inplace return it for you instead. (`auto` is nice.)
 *
 * You can generally just pretend this is an st. But not for wgmma's.
 */
template<
    typename _T,
    int _underlying_height,
    int _underlying_width,
    ducks::st_layout::all _layout,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile {
    using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
    using layout = _layout;
    using dtype = _T;

    static constexpr int underlying_height        = _underlying_height;
    static constexpr int underlying_width         = _underlying_width;
    static constexpr int underlying_rows          = underlying_height * kittens::TILE_DIM;
    static constexpr int underlying_cols          = underlying_width  * kittens::TILE_DIM;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int height              = _subtile_height;
    static constexpr int width               = _subtile_width;
    static constexpr int rows                = height * kittens::TILE_DIM;
    static constexpr int cols                = width  * kittens::TILE_DIM;
    static constexpr int num_elements        = rows * cols;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(dtype *src, int _row_offset, int _col_offset) {
        data = src;
        row_offset = _row_offset;
        col_offset = _col_offset;
    }

    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return data[detail::shared_indexer<underlying_height, underlying_width, layout>::idx(
            rowcol.x+row_offset, rowcol.y+col_offset
        )];
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return data[detail::shared_indexer<underlying_height, underlying_width, layout>::idx(
            rowcol.x+row_offset, rowcol.y+col_offset
        )];
    }

    // single-index operator[] is left undefined as it would likely be an improper use of st_subtile type

    // vector types
    using col_vec = sv<dtype, height>;
    using row_vec = sv<dtype, width>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace st {

/**
* @brief Concept for all shared tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as st::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::st::identifier

} // namespace st
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _height, int _width, ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf = st<bf16, _height, _width, layout>; // prelim tests indicate this is fastest default

template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_1x1 = st_bf<1, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_1x2 = st_bf<1, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_1x4 = st_bf<1, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_1x8 = st_bf<1, 8, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_2x1 = st_bf<2, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_2x2 = st_bf<2, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_2x4 = st_bf<2, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_4x1 = st_bf<4, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_4x2 = st_bf<4, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_4x4 = st_bf<4, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::naive> using st_bf_8x1 = st_bf<8, 1, layout>;

}
