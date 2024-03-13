#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layouts.cuh"
#include "rt_base.cuh"

namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */
    
// these are helper structs for type inference in vector maps
struct rt_id {};
struct rt_col_vec_id {};
struct rt_row_vec_id {};

// base register tile is 16x16
// row major flag will help compiler catch orientation errors during mma and whatnot
template<packed_type T2, int _height, int _width, rt_layout _layout=rt_row_layout>
struct rt {
    using identifier = rt_id;
    using layout = _layout;
    using dtype = T2;

    static constexpr int height              = _height;
    static constexpr int width               = _width;
    static constexpr int rows                = height  * rt_base<dtype, layout>::tile_size;
    static constexpr int cols                = width * rt_base<dtype, layout>::tile_size;
    static constexpr int tile_size           = rt_base<dtype, layout>::tile_size;
    static constexpr int num_elements        = rt_base<dtype, layout>::num_elements        * width * height;
    static constexpr int elements_per_thread = rt_base<dtype, layout>::elements_per_thread * width * height;
    static constexpr int packed_per_thread   = rt_base<dtype, layout>::packed_per_thread   * width * height;
    static constexpr int packed_per_tile     = rt_base<dtype, layout>::packed_per_thread; // bring that up

    rt_base<dtype, layout> tiles[height][width]; // should be row-major since we broadly expect row-major inputs

    // (assuming row_layout; if col_major then these are swapped)
    // relies on T2 being a packed type due to the particular layout demanded by mma.
    // essentially, each thread handles part of two separate rows in each 16x16 tile_base
    // correspondingly, we use a packed type where the .x is the top row and .y the bottom.
    struct col_vec {
        using identifier = rt_col_vec_id;
        using layout = layout;
        using dtype = dtype;

        rt_base<dtype, layout>::col_type data[height];

        static constexpr int outer_dim = height;
        static constexpr int inner_dim = sizeof(data[0])/sizeof(data[0][0]);

        __device__ inline       rt_base<dtype, layout>::col_type& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const rt_base<dtype, layout>::col_type& operator[](size_t idx) const { return data[idx]; }
    };
    // in the case of a column reduction, each thread contains data from 4 columns, so in
    // eager mode it will store a bit of all of them.
    struct row_vec {
        using identifier = rt_row_vec_id;
        using layout = layout;
        using dtype = dtype;

        rt_base<dtype, layout>::row_type data[width];

        static constexpr int outer_dim = width;
        static constexpr int inner_dim = sizeof(data[0])/sizeof(data[0][0]);

        __device__ inline       rt_base<dtype, layout>::row_type& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const rt_base<dtype, layout>::row_type& operator[](size_t idx) const { return data[idx]; }
    };

    /* ----------  SUBTILE  ---------- */

    template<int subtile_height>
    __device__ inline rt<dtype, subtile_height, width, layout> &subtile_inplace(int idx) {
        static_assert(height % subtile_height == 0, "subtile height should evenly divide tile height.");
        return reinterpret_cast<rt<dtype, subtile_height, width, layout>&>(
            tiles[idx*subtile_height]
        );
    }
};

/* ----------  CONCEPTS  ---------- */

template<typename T> concept rt_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_id>; // Checks if T::dentifier is abstract_rt
// specialized types for specialized functions
template<typename T>
concept rt_type_rowlayout = rt_type<T> && std::is_same_v<typename T::layout, rt_row_layout>;
template<typename T>
concept rt_type_collayout = rt_type<T> && std::is_same_v<typename T::layout, rt_col_layout>;

template<typename T>
concept rt_col_vec_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_col_vec_id>; // Checks if T::identifier is abstract_vector
template<typename T>
concept rt_row_vec_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_row_vec_id>; // Checks if T::identifier is abstract_vector
template<typename T>
concept rt_vec_type = rt_row_vec_type<T> || rt_col_vec_type<T>;



/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _height, int _width, rt_layout layout=rt_row_layout> using rt_fl = rt<float2, _height, _width, layout>;
template<int _height, int _width, rt_layout layout=rt_row_layout> using rt_bf = rt<bf16_2, _height, _width, layout>;

// layout, type, and size wrappers
// sizes are chosen with the assumption that you aren't going to want to fit more than
// 8 subtiles on a thread. (if you need more, you can of course add your own using's.)

template<rt_layout layout=rt_row_layout> using rt_fl_1x1 = rt_fl<1, 1, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x2 = rt_fl<1, 2, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x4 = rt_fl<1, 4, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x8 = rt_fl<1, 8, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x1 = rt_fl<2, 1, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x2 = rt_fl<2, 2, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x4 = rt_fl<2, 4, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_4x1 = rt_fl<4, 1, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_4x2 = rt_fl<4, 2, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_8x1 = rt_fl<8, 1, layout>; // 64 registers used

template<rt_layout layout=rt_row_layout> using rt_bf_1x1 = rt_bf<1, 1, layout>; //  4 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x2 = rt_bf<1, 2, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x4 = rt_bf<1, 4, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x8 = rt_bf<1, 8, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x1 = rt_bf<2, 1, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x2 = rt_bf<2, 2, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x4 = rt_bf<2, 4, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_4x1 = rt_bf<4, 1, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_4x2 = rt_bf<4, 2, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_8x1 = rt_bf<8, 1, layout>; // 32 registers used

}