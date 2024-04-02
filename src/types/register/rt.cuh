/**
 * @file
 * @brief The main ThunderKittens register tile struct, where most computation happens.
 */

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
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register tiles live.
 */
namespace rt {
/**
 * @brief A dummy type used to identify register tiles.
 * 
 * For a type to quack like an rt, it should define its identifier as ducks::rt::identifier.
 * If a type quacks like ducks::rt::identifier, it will be treated as an rt by compiler checks.
 */
struct identifier {};
} // namespace rt
} // namespace ducks

/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal base tiles, either row-major or column-major.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles. It supports both
 * row-major and column-major layouts and includes helper structs for type inference
 * in vector maps.
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< Type identifier for the rt structure.
    using layout = _layout; ///< Layout of the matrix tile.
    using dtype = T2; ///< Data type of the matrix elements.

    static constexpr int height              = _height; ///< Height in subtiles.
    static constexpr int width               = _width; ///< Width in subtiles.
    static constexpr int rows                = height  * rt_base<dtype, layout>::tile_size; ///< Total number of rows.
    static constexpr int cols                = width * rt_base<dtype, layout>::tile_size; ///< Total number of columns.
    static constexpr int tile_size           = rt_base<dtype, layout>::tile_size; ///< Size of the base tile.
    static constexpr int num_elements        = rt_base<dtype, layout>::num_elements        * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<dtype, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread   = rt_base<dtype, layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile     = rt_base<dtype, layout>::packed_per_thread; ///< Packed elements per tile.

    rt_base<dtype, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

    using col_vec = rv<dtype, height, rt_base<dtype, layout>::col_vec_pack>; ///< A type representing a column vector for this tile.
    using row_vec = rv<dtype, width , rt_base<dtype, layout>::row_vec_pack>; ///< A type representing a column vector for this tile.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt {
/**
* @brief Concept for all register tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
/**
* @brief Concept for register tiles with row layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has a nested type layout that is ducks::rt_layout::row.
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
* @brief Concept for register tiles with col layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has a nested type layout that is ducks::rt_layout::col.
*/
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
// 8 subtiles on a warp. (Could be wrong!)

///  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x1 = rt_fl<1, 1, layout>;
/// 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x2 = rt_fl<1, 2, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x4 = rt_fl<1, 4, layout>;
/// 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_1x8 = rt_fl<1, 8, layout>;
/// 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x1 = rt_fl<2, 1, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x2 = rt_fl<2, 2, layout>;
/// 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_2x4 = rt_fl<2, 4, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_4x1 = rt_fl<4, 1, layout>;
/// 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_4x2 = rt_fl<4, 2, layout>;
/// 64 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl_8x1 = rt_fl<8, 1, layout>;

///  4 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x1 = rt_bf<1, 1, layout>;
///  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x2 = rt_bf<1, 2, layout>;
/// 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x4 = rt_bf<1, 4, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_1x8 = rt_bf<1, 8, layout>;
///  8 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x1 = rt_bf<2, 1, layout>;
/// 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x2 = rt_bf<2, 2, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_2x4 = rt_bf<2, 4, layout>;
/// 16 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_4x1 = rt_bf<4, 1, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_4x2 = rt_bf<4, 2, layout>;
/// 32 registers used
template<ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf_8x1 = rt_bf<8, 1, layout>;

} // namespace kittens
