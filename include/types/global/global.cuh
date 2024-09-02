/**
 * @file
 * @brief An aggregate header file for all the global types defined by ThunderKittens.
 */

#pragma once

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif
#include "util.cuh"
#include "gvl.cuh"
#include "gtl.cuh"

// Some helper checks for convenience
namespace kittens {
namespace ducks {
namespace g {
namespace detail {
template<typename L, typename T> struct check_dims {};
template<ducks::gt::l::all L, ducks::st::all ST> struct check_dims<L, ST> {
    static_assert(std::is_same_v<typename L::T, typename ST::T>, "SKILL ISSUE: tile type does not match that of its global tile layout descriptor");
    static_assert(L::base_height == ST::height, "SKILL ISSUE: tile height does not match that of its global tile layout descriptor");
    static_assert(L::base_width  == ST::width,  "SKILL ISSUE: tile width does not match that of its global tile layout descriptor");
};
template<ducks::gv::l::all L, ducks::sv::all SV> struct check_dims<L, SV> {
    static_assert(std::is_same_v<typename L::T, typename SV::T>, "SKILL ISSUE: vector type does not match that of its global vector layout descriptor");
    static_assert(L::base_tiles == SV::tiles,  "SKILL ISSUE: vector length does not match that of its global vector layout descriptor");
};
template<typename T> struct check_tma {
    static_assert(T::tma, "SKILL ISSUE: passed a non-TMA global descriptor to a function that requires a TMA descriptor");
};
template<typename T> struct check_raw {
    static_assert(T::raw, "SKILL ISSUE: passed a TMA-only global descriptor to a function that requires raw global memory");
};
}
template<typename L, typename U> struct check_tma {
    detail::check_dims<L, U> _1;
    detail::check_tma<L> _2;
};
template<typename L, typename U> struct check_raw {
    detail::check_dims<L, U> _1;
    detail::check_raw<L> _2;
};
}
}
}