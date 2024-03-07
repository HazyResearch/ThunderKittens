/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */
 
#pragma once

#include <type_traits>
#include "../../common/base_types.cuh"
#include "rt_layouts.cuh"
#include "../../common/common.cuh"

namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

/**
 * @brief Identifier for the base register tile type.
 */
struct rt_base_id {};

/**
 * @brief Base structure for a 16x16 register tile.
 * 
 * This structure defines the base type for register tiles used in matrix operations.
 * It includes type definitions and constants for the size of the tile, the number of elements,
 * and the layout of the data within the tile.
 *
 * @tparam T2 The packed data type for the elements of the tile.
 * @tparam _layout The layout of the matrix (row-major or column-major).
 */
template<packed_type T2, rt_layout _layout>
struct rt_base {
    using identifier = rt_base_id; ///< Type identifier for the base register tile.
    using layout = _layout;        ///< Layout of the matrix.
    using dtype = T2;              ///< Data type of the elements.

    static constexpr int tile_size            = 16; ///< Size of one side of the tile (16 elements).
    static constexpr int num_elements         = tile_size*tile_size; ///< Total number of elements in the tile (256).
    static constexpr int elements_per_thread  = num_elements / 32;   ///< Number of elements held by each thread (8).

    // Static assert to ensure that packing<T2>::num() is valid
    static_assert(kittens::base_types::packing<T2>::num() == 2, "T2 must be a packed type");

    static constexpr int packed_per_thread    = elements_per_thread / kittens::base_types::packing<T2>::num(); ///< Number of packed elements per thread.
    static constexpr int registers_per_thread = packed_per_thread * sizeof(T2) / 4; ///< Number of 32-bit registers needed per thread.

    // Using an array type for both makes the access a bit more regular, which simplifies rt_vector.cuh
    // All this should be hidden from the user, anyways.
    using col_type = std::conditional_t<layout::row, T2[1], T2[2]>; ///< Type for holding row reductions.
    using row_type = std::conditional_t<!layout::row, T2[1], T2[2]>; ///< Type for holding column reductions.

    T2 data[packed_per_thread]; ///< Data storage for the tile.
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Wrapper for a base register tile with float2 data type and specified layout.
 * 
 * @tparam L The layout of the matrix (row-major or column-major), defaults to row-major.
 */
template<typename L=rt_row_layout> using rt_base_fl = rt_base<float2, L>; // Note float2! Otherwise you will get bugs.

/**
 * @brief Wrapper for a base register tile with bf16_2 data type and specified layout.
 * 
 * @tparam L The layout of the matrix (row-major or column-major), defaults to row-major.
 */
template<typename L=rt_row_layout> using rt_base_bf = rt_base<bf16_2, L>;

// Explicit template instantiations
template struct rt_base<float2, rt_row_layout>;
template struct rt_base<bf16_2, rt_row_layout>;

}
