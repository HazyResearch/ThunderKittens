#pragma once

#include "../../common/common.cuh"

#include "st_layouts.cuh"

// I need to redo this to fix the layout and also make it pretty.
// However, I want something stupidly simple that I can test register ops with.
// So, this is going to be the starting point for now.

namespace kittens {

struct st_id {};
struct st_col_vec_id {};
struct st_row_vec_id {};

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile.
 * @tparam _height The height of the tile in terms of 16-element vectors.
 * @tparam _width The width of the tile in terms of 16-element vectors.
 * @tparam _layout The memory layout of the tile.
 */
template<typename T, int _height, int _width, typename _layout>
struct st {
    using identifier = st_id; ///< Type identifier for shared memory tile.
    using layout = _layout; ///< Memory layout of the tile.
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int height              = _height; ///< Height of the tile in terms of 16-element vectors.
    static constexpr int width               = _width; ///< Width of the tile in terms of 16-element vectors.
    static constexpr int rows                = height * 16; ///< Total number of rows in the tile.
    static constexpr int cols                = width  * 16; ///< Total number of columns in the tile.
    static constexpr int num_elements        = width  * height * 16*16; ///< Total number of elements in the tile.
    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)
    static_assert(!std::is_same_v<_layout, kittens::st_wgmma_row_128b_layout> || height >= 2); // if using 128-byte swizzling, needs at least 32 rows.
    static_assert(!std::is_same_v<_layout, kittens::st_wgmma_col_128b_layout> || width >= 2); // or cols

    dtype data[rows*cols]; ///< Raw data storage for the tile.

    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return data[detail::st_idx<_layout>(rowcol.x, rowcol.y, height, width)];
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return data[detail::st_idx<_layout>(rowcol.x, rowcol.y, height, width)];
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }
    template<typename _layout, typename = std::enable_if_t<kittens::st_wgmma_layout<_layout>::value>>
    __device__ inline uint64_t descriptor(int chunk_idx=0) const {
        uint64_t desc = 0x0000000000000000;
        uint64_t start_addr;
        if constexpr (kittens::st_wgmma_row_layout<_layout>) { // we're in a row layout mode, so the next chunk we want is by column.
            start_addr = (uint64_t)&(*this)[{0, 16*chunk_idx}];
        }
        else { // we're in a column layout mode, so the next chunk we want is by row.
            start_addr = (uint64_t)&(*this)[{16*chunk_idx, 0}];
        }
        desc |= matrix_descriptor_encode(start_addr);
        desc |= matrix_descriptor_encode((uint64_t)128) << 16;
        desc |= matrix_descriptor_encode((uint64_t)256) << 32;
        uint64_t base_offset = 0;
        if constexpr (_layout::swizzling_mode == 1) {
            if((uint64_t)(start_addr) % 256 != 0) {
                base_offset = (start_addr >> 0x7) ^ 0x7;
            }
        }
        if constexpr (_layout::swizzling_mode == 2) {
            if((uint64_t)(start_addr) % 512 != 0) {
                base_offset = (start_addr >> 0x7) ^ 0x7;
            }
        }
        if constexpr (_layout::swizzling_mode == 3) {
            if((uint64_t)(start_addr) % 1024 != 0) {
                base_offset = (start_addr >> 0x7) ^ 0x7;
            }
        }
        desc |= ((uint64_t)base_offset) << 49;
        desc |= ((uint64_t)_layout::swizzling_mode) << 62;
        return desc;
    }

    // Vector types for columns and rows.
    struct col_vec {
        using identifier = st_col_vec_id; ///< Type identifier for column vector.
        using dtype = T; ///< Data type of the elements in the vector.
        static constexpr int length = rows; ///< Length of the column vector.

        dtype data[length]; ///< Raw data storage for the column vector.

        __device__ inline       dtype& operator[](const int &rowcol)       { return data[rowcol]; }
        __device__ inline const dtype& operator[](const int &rowcol) const { return data[rowcol]; }
    };
    struct row_vec {
        using identifier = st_row_vec_id; ///< Type identifier for row vector.
        using dtype = T; ///< Data type of the elements in the vector.
        static constexpr int length = cols; ///< Length of the row vector.

        dtype data[length]; ///< Raw data storage for the row vector.

        __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
    };
};

template<typename T>
struct is_st_type {
    static constexpr bool value = std::is_same<typename T::identifier, st_id>::value;
};

template<typename T>
struct is_st_col_vec_type {
    static constexpr bool value = std::is_same<typename T::identifier, st_col_vec_id>::value;
};

template<typename T>
struct is_st_row_vec_type {
    static constexpr bool value = std::is_same<typename T::identifier, st_row_vec_id>::value;
};

template<typename T>
struct is_st_vec_type {
    static constexpr bool value = is_st_col_vec_type<T>::value || is_st_row_vec_type<T>::value;
};

template<typename T>
using st_type = std::enable_if_t<is_st_type<T>::value, bool>;

template<typename T>
using st_col_vec_type = std::enable_if_t<is_st_col_vec_type<T>::value, bool>;

template<typename T>
using st_row_vec_type = std::enable_if_t<is_st_row_vec_type<T>::value, bool>;

template<typename T>
using st_vec_type = std::enable_if_t<is_st_vec_type<T>::value, bool>;

// Provided templates for creating instances of `st` with specific data types and layouts.
template<int _height, int _width, typename layout> using st_bf = st<bf16, _height, _width, layout>; // prelim tests indicate this is fastest default

template<typename layout> using st_bf_1x1 = st_bf<1, 1, layout>;
template<typename layout> using st_bf_1x2 = st_bf<1, 2, layout>;
template<typename layout> using st_bf_1x4 = st_bf<1, 4, layout>;
template<typename layout> using st_bf_1x8 = st_bf<1, 8, layout>;

template<typename layout> using st_bf_2x1 = st_bf<2, 1, layout>;
template<typename layout> using st_bf_2x2 = st_bf<2, 2, layout>;
template<typename layout> using st_bf_2x4 = st_bf<2, 4, layout>;

template<typename layout> using st_bf_4x1 = st_bf<4, 1, layout>;
template<typename layout> using st_bf_4x2 = st_bf<4, 2, layout>;
template<typename layout> using st_bf_4x4 = st_bf<4, 4, layout>;

template<typename layout> using st_bf_8x1 = st_bf<8, 1, layout>;


// Vector types for shared memory tiles.
template<typename layout> using sv_bf_row_1 = typename st<bf16,1,1,layout>::row_vec;
template<typename layout> using sv_bf_row_4 = typename st<bf16,1,4,layout>::row_vec;

template<typename layout> using sv_bf_col_1 = typename st<bf16,1,1,layout>::col_vec;
template<typename layout> using sv_bf_col_4 = typename st<bf16,4,1,layout>::col_vec;

template<typename layout> using sv_fl_row_1 = typename st<float,1,1,layout>::row_vec;
template<typename layout> using sv_fl_row_4 = typename st<float,1,4,layout>::row_vec;

template<typename layout> using sv_fl_col_1 = typename st<float,1,1,layout>::col_vec;
template<typename layout> using sv_fl_col_4 = typename st<float,4,1,layout>::col_vec;


}
