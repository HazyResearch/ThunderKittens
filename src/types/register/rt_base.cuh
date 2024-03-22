#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"

namespace kittens {

/** 
 * @brief Base namespace for duck-related data structures and operations.
 */
namespace ducks {

/** 
 * @brief Namespace for base register tile operations.
 */
namespace rt_base {

/** 
 * @brief Identifier struct for rt_base namespace.
 */
struct identifier {};

} // namespace rt_base
} // namespace ducks

/** 
 * @brief Template struct for base register tile which is 16x16.
 * 
 * @tparam T2 Data type for the elements.
 * @tparam _layout Layout type defining row or column major format.
 */
template<typename T2, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< Identifier for the rt_base struct.
    using layout = _layout; ///< Layout type (row or column major).
    using dtype = T2; ///< Data type of the elements.

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2>,
        "rt_base was provided an unsupported type."
    );

    static constexpr int tile_size            = 16; ///< Tile size for the base register tile.
    static constexpr int num_elements         = tile_size*tile_size; ///< Total number of elements in the tile.
    static constexpr int elements_per_thread  = num_elements / 32; ///< Number of elements per thread.

    static constexpr int packed_per_thread    = elements_per_thread / base_types::packing<T2>::num(); ///< Number of packed elements per thread.
    static constexpr int registers_per_thread = packed_per_thread * sizeof(T2) / 4; ///< Number of registers per thread (32-bit words).

    // using an array type for both makes the access a bit more regular, which simplifies rt_vector.cuh
    // all this should be hidden from the user, anyways.
    static constexpr int col_vec_pack = layout::is_row ? 1 : 2; ///< Column vector pack size for row reductions.
    static constexpr int row_vec_pack = layout::is_row ? 2 : 1; ///< Row vector pack size for column reductions.

    T2 data[packed_per_thread]; ///< Storage for the packed data.
};

/** 
 * @brief Wrapper for rt_base with float2 data type and default row layout.
 * 
 * @tparam L Layout type (defaults to row layout).
 */
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float2, L>; // Note float2! Otherwise you will get bugs.

/** 
 * @brief Wrapper for rt_base with bf16_2 data type and default row layout.
 * 
 * @tparam L Layout type (defaults to row layout).
 */
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16_2, L>;

} // namespace kittens
