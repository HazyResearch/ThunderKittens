/**
 * @file
 * @brief Layouts and manipulations for register tiles.
 */
 
#pragma once

#include <type_traits>

namespace kittens {

/**
 * @brief Represents a row-major layout for matrices.
 */
struct rt_row_layout { static constexpr bool row=true; };

/**
 * @brief Represents a column-major layout for matrices, typically used for the B-matrix in MMA operations.
 */
struct rt_col_layout { static constexpr bool row=false; };

/**
 * @brief Structure that checks if a type represents a matrix layout.
 * 
 * @tparam T The type to check against the rt_row_layout and rt_col_layout.
 */
template<typename T>
concept rt_layout = (
    std::is_same_v<T, rt_row_layout>  ||
    std::is_same_v<T, rt_col_layout>
);

/**
 * @brief Determines the transpose of a given layout type.
 * 
 * @tparam L The layout type to transpose.
 */
template<typename L> struct transpose_layout                { using type = rt_col_layout; };

/**
 * @brief Specialization of transpose_layout for rt_col_layout.
 */
template<>            struct transpose_layout<rt_col_layout> { using type = rt_row_layout; };

}
