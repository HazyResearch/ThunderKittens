/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */

#pragma once

#include "../../common/common.cuh"
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
 * This is particularly useful for subtiles.
 */
struct identifier {};
} // namespace st
namespace st_layout {
struct swizzled {};
struct wmma {};
} // namespace st_layout
} // namespace ducks

// Forward declaration of subtile
template<
    typename _T,
    int _underlying_height,
    int _underlying_width,
    int _subtile_height,
    int _subtile_width,
    typename _layout = ducks::st_layout::swizzled
>
struct st_subtile;

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 */
template<typename _T, int _rows, int _cols, typename _layout = ducks::st_layout::swizzled>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    using layout = _layout;

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows          = _rows;
    static constexpr int underlying_cols          = _cols;
    static constexpr int underlying_height        = _rows / kittens::TILE_DIM;
    static constexpr int underlying_width         = _cols / kittens::TILE_DIM;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static_assert(rows % TILE_DIM == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static_assert(cols % TILE_DIM == 0, "Cols must be divisible by the tile dimension");
    static constexpr int height              = _rows / kittens::TILE_DIM; ///< Height of the tile in terms of 16-element subtiles.
    static constexpr int width               = _cols / kittens::TILE_DIM; ///< Width of the tile in terms of 16-element subtiles.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)
    static_assert(std::is_same_v<_layout, ducks::st_layout::swizzled> || std::is_same_v<_layout, ducks::st_layout::wmma>);

    static constexpr int swizzle_bytes = (
        sizeof(dtype) == 2 ? (
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            underlying_width%2 == 0 ? 128 : 64
        ) : -1
    );

    // wgmma layout with swizzling
    dtype data[rows*cols]; ///< Raw data storage for the tile.

    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major coord default
    if constexpr (std::is_same_v<_layout, ducks::st_layout::swizzled>) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swizzle);
        }
        else {
            constexpr int minicol_width = 32/sizeof(T);
            int supercol = c/minicol_width, subcol = c%minicol_width;
            return &ptr[(supercol*underlying_rows + r)*minicol_width + subcol];
        }
    }
    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled layouts.
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol.x, rowcol.y);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol.x, rowcol.y);
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    // vector types
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<
        dtype, rows, cols, subtile_rows, subtile_cols
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
    int _underlying_rows,
    int _underlying_cols,
    int _subtile_rows,
    int _subtile_cols,
    typename _layout
>
struct st_subtile {
    using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    using layout = _layout;

    static constexpr int underlying_rows          = _underlying_rows;
    static_assert(underlying_rows % TILE_DIM == 0, "Underlying rows must be divisible by the tile dimension");
    static constexpr int underlying_cols          = _underlying_cols;
    static_assert(underlying_cols % TILE_DIM == 0, "Underlying cols must be divisible by the tile dimension");
    static constexpr int underlying_height        = underlying_rows / kittens::TILE_DIM;
    static constexpr int underlying_width         = underlying_cols / kittens::TILE_DIM;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _subtile_rows;
    static_assert(rows % TILE_DIM == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _subtile_cols;
    static_assert(cols % TILE_DIM == 0, "Cols must be divisible by the tile dimension");
    static constexpr int height              = rows / kittens::TILE_DIM;
    static constexpr int width               = cols / kittens::TILE_DIM;
    static constexpr int num_elements        = rows * cols;

    static constexpr int swizzle_bytes = (
        sizeof(dtype) == 2 ? (
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            underlying_width%2 == 0 ? 128 : 64
        ) : -1
    );

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(dtype *src, int _row_offset, int _col_offset) {
        data = src;
        row_offset = _row_offset;
        col_offset = _col_offset;
    }

    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major coord default
        if constexpr (std::is_same_v<_layout, ducks::st_layout::swizzled>) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swizzle);
        }
        else {
            constexpr int minicol_width = 32/sizeof(dtype);
            int supercol = c/minicol_width, subcol = c%minicol_width;
            return &ptr[(supercol*underlying_rows + r)*minicol_width + subcol];
        }
    }
    /**
     * @brief Access a shared tile element using a row and column, as if the tile were row-major.
     *
     * This is the preferred way to access memory within a shared tile, which abstracts
     * indexing calculations for swizzled layouts.
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol.x+row_offset, rowcol.y+col_offset);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol.x+row_offset, rowcol.y+col_offset);
    }

    // single-coord operator[] is left undefined as it would likely be an improper use of st_subtile type.
    // can of course be end-run by just accessing .data directly.

    // vector types
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
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

template<int _height, int _width, typename L=ducks::st_layout::swizzled> using st_bf = st<bf16,  _height, _width, L>;
template<int _height, int _width, typename L=ducks::st_layout::swizzled> using st_hf = st<half,  _height, _width, L>;
template<int _height, int _width, typename L=ducks::st_layout::swizzled> using st_fl = st<float, _height, _width, L>;

}
