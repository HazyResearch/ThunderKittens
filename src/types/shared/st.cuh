#pragma once

#include "../../common/common.cuh"
#include "st_layout.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
namespace st {
struct identifier {};
} // namespace st
namespace sv {
struct col_vec_identifier {};
struct row_vec_identifier {};
} // namespace sv
} // namespace ducks

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

    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

    __device__ inline uint64_t descriptor(int chunk_idx=0) const {
        uint64_t desc = 0x0000000000000000;
        uint64_t start_addr;
        if constexpr (ducks::st_layout::wgmma_row<layout>) { // we're in a row layout mode, so the next chunk we want is by column.
            start_addr = (uint64_t)&(*this)[{0, 16*chunk_idx}];
        }
        else { // we're in a column layout mode, so the next chunk we want is by row.
            start_addr = (uint64_t)&(*this)[{16*chunk_idx, 0}];
        }
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

    // vector types
    struct col_vec {
        using identifier = ducks::sv::col_vec_identifier;
        using dtype = _T;
        static constexpr int length = rows;

        dtype data[length];

        __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
    };
    struct row_vec {
        using identifier = ducks::sv::row_vec_identifier;
        using dtype = _T;
        static constexpr int length = cols;

        dtype data[length];

        __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
    };

    template<int _subtile_height, int _subtile_width>
    struct subtile_t {
        using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
        using layout = _layout;
        using dtype = _T;

        static constexpr int underlying_height   = _height;
        static constexpr int underlying_width    = _width;
        static constexpr int underlying_rows     = underlying_height * 16;
        static constexpr int underlying_cols     = underlying_width  * 16;
        static constexpr int num_elements        = underlying_height * underlying_width * 16*16;

        static constexpr int height              = _subtile_height;
        static constexpr int width               = _subtile_width;
        static constexpr int rows                = height * 16;
        static constexpr int cols                = width  * 16;

        dtype *data;
        int row_offset, col_offset;

        __device__ subtile_t(dtype *src, int _row_offset, int _col_offset): data(src), row_offset(_row_offset), col_offset(_col_offset) {} // constructor

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

        // vector types
        struct col_vec {
            using identifier = ducks::sv::col_vec_identifier;
            using dtype = _T;
            static constexpr int length = rows;

            dtype data[length];

            __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
            __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
        };
        struct row_vec {
            using identifier = ducks::sv::row_vec_identifier;
            using dtype = _T;
            static constexpr int length = cols;

            dtype data[length];

            __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
            __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
        };
    };

    template<int subtile_height, int subtile_width>
    __device__ inline subtile_t<subtile_height, subtile_width> subtile(int tile_row_offset, int tile_col_offset) {
        return subtile_t<subtile_height, subtile_width>(&data[0], subtile_height*16*tile_row_offset, subtile_width*16*tile_col_offset);
    }
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

namespace sv {

template<typename T>
concept col_vec = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, col_vec_identifier>; // Checks if T::identifier is abstract_vector

template<typename T>
concept row_vec = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, row_vec_identifier>; // Checks if T::identifier is abstract_vector
template<typename T>
concept all = col_vec<T> || row_vec<T>;

} // namespace sv
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// tile types
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

// vector types
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_bf_row_1 = st<bf16,1,1,layout>::row_vec;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_bf_row_4 = st<bf16,1,4,layout>::row_vec;

template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_bf_col_1 = st<bf16,1,1,layout>::col_vec;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_bf_col_4 = st<bf16,4,1,layout>::col_vec;

template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_fl_row_1 = st<float,1,1,layout>::row_vec;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_fl_row_4 = st<float,1,4,layout>::row_vec;

template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_fl_col_1 = st<float,1,1,layout>::col_vec;
template<ducks::st_layout::all layout=ducks::st_layout::xor_swizzle> using sv_fl_col_4 = st<float,4,1,layout>::col_vec;

}
