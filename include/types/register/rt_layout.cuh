/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rt_layout
 * 
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace rt_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct row { static constexpr bool is_row=true;  }; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a register tile.
 */
struct col { static constexpr bool is_row=false; }; // for the B-matrix of MMA ops.

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

/**
 * @brief A struct to generate a transposed layout.
 */
template<all L> struct transpose      { using type = col; };
template<>      struct transpose<col> { using type = row; };

} // namespace ducks::rt_layout::all
} // namespace ducks
} // namespace kittens
