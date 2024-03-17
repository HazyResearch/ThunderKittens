#pragma once

#include <concepts>

namespace kittens {
namespace concepts {

struct rt_row_layout { static constexpr bool row=true;  }; // for most matrices
struct rt_col_layout { static constexpr bool row=false; }; // for the B-matrix of MMA ops.

template<typename T>
concept rt_layout = (
    std::is_same_v<T, rt_row_layout>  ||
    std::is_same_v<T, rt_col_layout>
);

template<rt_layout L> struct transpose_layout                { using type = rt_col_layout; };
template<>            struct transpose_layout<rt_col_layout> { using type = rt_row_layout; };

} // namespace concepts
} // namespace kittens
