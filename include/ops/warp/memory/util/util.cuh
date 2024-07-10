/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

// template magic allows arrays of these objects to be copied, too.
namespace detail {
template<typename T, uint32_t... dims> struct size_info;
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
template<typename T, uint32_t dim, uint32_t... rest_dims> struct size_info<T, dim, rest_dims...> {
    static constexpr uint32_t elements = dim*size_info<T, rest_dims...>::elements;
    static constexpr uint32_t bytes    = dim*size_info<T, rest_dims...>::bytes;
};
}
template<typename T, uint32_t... dims> constexpr uint32_t size_elements = detail::size_info<T, dims...>::elements;
template<typename T, uint32_t... dims> constexpr uint32_t size_bytes    = detail::size_info<T, dims...>::bytes;

}

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#include "dsmem.cuh"
#endif