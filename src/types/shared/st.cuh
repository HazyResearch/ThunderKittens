#pragma once

#include "../../common/common.cuh"
#include "st_layout.cuh"
#include "sv.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
namespace st {
struct identifier {};
} // namespace st
} // namespace ducks

// wgmma helpers
namespace detail {
// see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template<ducks::st_layout::wgmma layout>
__device__ inline uint64_t matrix_descriptor(uint64_t start_addr) {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(start_addr);
    desc |= matrix_descriptor_encode((uint64_t)128) << 16;
    desc |= matrix_descriptor_encode((uint64_t)256) << 32;
    uint64_t base_offset = 0;
    if constexpr (layout::swizzling_mode == 3) {
        if((uint64_t)(start_addr) % 256 != 0) {
            base_offset = (start_addr >> 0x7) & 0x7;
        }
    }
    if constexpr (layout::swizzling_mode == 2) {
        if((uint64_t)(start_addr) % 512 != 0) {
            base_offset = (start_addr >> 0x7) & 0x7;
        }
    }
    if constexpr (layout::swizzling_mode == 1) {
        if((uint64_t)(start_addr) % 1024 != 0) {
            base_offset = (start_addr >> 0x7) & 0x7;
        }
    }
    desc |= ((uint64_t)base_offset) << 49;
    desc |= ((uint64_t)layout::swizzling_mode) << 62;
    return desc;
}
}

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

// NOT expecting a packed type
template<typename _T, int _height, int _width, ducks::st_layout::all _layout>
struct st {
    using identifier = ducks::st::identifier;
    using layout = _layout;
    using dtype = _T;

    static constexpr int height              = _height;
    static constexpr int width               = _width;
    static constexpr int rows                = height * 16;
    static constexpr int cols                = width  * 16;
    static constexpr int num_elements        = width  * height * 16*16;

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    static_assert(
        !std::is_same_v<layout, ducks::st_layout::tma_swizzle> || width == 1 || width == 2 || width == 4,
        "For TMA swizzled modes, shared tile width must be 1, 2, or 4."
    ); // TMA swizzling only appears to work with a few particular layout dimensions.

    // wgmma layout with swizzling
    dtype data[rows*cols];

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

    __device__ inline uint64_t descriptor(int chunk_idx=0) const {
        uint64_t start_addr;
        if constexpr (ducks::st_layout::wgmma_row<layout>) { // we're in a row layout mode, so the next chunk we want is by column.
            start_addr = (uint64_t)&(*this)[{0, 16*chunk_idx}];
        }
        else { // we're in a column layout mode, so the next chunk we want is by row.
            start_addr = (uint64_t)&(*this)[{16*chunk_idx, 0}];
        }
        return detail::matrix_descriptor<layout>(start_addr);
    }

    // vector types
    using col_vec = sv<dtype, height>;
    using row_vec = sv<dtype, width>;
    template<int subtile_height, int subtile_width> using subtile = st_subtile<
        dtype, height, width, layout,
        subtile_height, subtile_width
    >;
};


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
    static constexpr int underlying_rows          = underlying_height * 16;
    static constexpr int underlying_cols          = underlying_width  * 16;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int height              = _subtile_height;
    static constexpr int width               = _subtile_width;
    static constexpr int rows                = height * 16;
    static constexpr int cols                = width  * 16;
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

    __device__ inline uint64_t descriptor(int chunk_idx=0) const {
        uint64_t start_addr;
        if constexpr (ducks::st_layout::wgmma_row<layout>) { // we're in a row layout mode, so the next chunk we want is by column.
            start_addr = (uint64_t)&(*this)[{0, 16*chunk_idx}];
        }
        else { // we're in a column layout mode, so the next chunk we want is by row.
            start_addr = (uint64_t)&(*this)[{16*chunk_idx, 0}];
        }
        return detail::matrix_descriptor<layout>(start_addr);
    }

    // vector types
    using col_vec = sv<dtype, height>;
    using row_vec = sv<dtype, width>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace st {

template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is st::identifier
template<typename T> concept row_layout = all<T> && ducks::st_layout::row<typename T::layout>;
template<typename T> concept col_layout = all<T> && ducks::st_layout::col<typename T::layout>;

} // namespace st
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _height, int _width, ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf = st<bf16, _height, _width, layout>; // prelim tests indicate this is fastest default

template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_1x1 = st_bf<1, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_1x2 = st_bf<1, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_1x4 = st_bf<1, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_1x8 = st_bf<1, 8, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_2x1 = st_bf<2, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_2x2 = st_bf<2, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_2x4 = st_bf<2, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_4x1 = st_bf<4, 1, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_4x2 = st_bf<4, 2, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_4x4 = st_bf<4, 4, layout>;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using st_bf_8x1 = st_bf<8, 1, layout>;

}
