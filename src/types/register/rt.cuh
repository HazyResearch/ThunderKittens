#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rv.cuh"

namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
namespace rt {
struct identifier {};
} // namespace rt
} // namespace ducks

// base register tile is 16x16
// row major flag will help compiler catch orientation errors during mma and whatnot
template<typename T2, int _height, int _width, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier;
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

    using col_vec = rv<dtype, height, rt_base<dtype, layout>::col_vec_pack>;
    using row_vec = rv<dtype, width , rt_base<dtype, layout>::row_vec_pack>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt {

template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is rt::identifier
// specialized types for specialized functions
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;

} // namespace rt
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _height, int _width, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float2, _height, _width, layout>;
template<int _height, int _width, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16_2, _height, _width, layout>;

// layout, type, and size wrappers
// sizes are chosen with the assumption that you aren't going to want to fit more than
// 8 subtiles on a thread. (if you need more, you can of course add your own using's.)

template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x1 = rt_fl<1, 1, layout>; //  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x2 = rt_fl<1, 2, layout>; // 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x4 = rt_fl<1, 4, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x8 = rt_fl<1, 8, layout>; // 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x1 = rt_fl<2, 1, layout>; // 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x2 = rt_fl<2, 2, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x4 = rt_fl<2, 4, layout>; // 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_4x1 = rt_fl<4, 1, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_4x2 = rt_fl<4, 2, layout>; // 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_8x1 = rt_fl<8, 1, layout>; // 64 registers used

template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x1 = rt_bf<1, 1, layout>; //  4 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x2 = rt_bf<1, 2, layout>; //  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x4 = rt_bf<1, 4, layout>; // 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x8 = rt_bf<1, 8, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x1 = rt_bf<2, 1, layout>; //  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x2 = rt_bf<2, 2, layout>; // 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x4 = rt_bf<2, 4, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_4x1 = rt_bf<4, 1, layout>; // 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_4x2 = rt_bf<4, 2, layout>; // 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_8x1 = rt_bf<8, 1, layout>; // 32 registers used

} // namespace kittens
