#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
namespace rt_layout {

struct row { static constexpr bool is_row=true;  }; // for most matrices
struct col { static constexpr bool is_row=false; }; // for the B-matrix of MMA ops.

template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

template<all L> struct transpose      { using type = col; };
template<>      struct transpose<col> { using type = row; };

} // namespace ducks::rt_layout::all
} // namespace ducks
} // namespace kittens
