/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {
namespace detail {

// template magic allows arrays of these objects to be copied, too.
template<typename T, uint32_t... dims> struct transfer_bytes;
template<ducks::st::all ST> struct transfer_bytes<ST> { static constexpr uint32_t bytes = ST::num_elements * sizeof(typename ST::dtype); };
template<ducks::sv::all SV> struct transfer_bytes<SV> { static constexpr uint32_t bytes = SV::length * sizeof(typename SV::dtype); };
template<typename T, uint32_t dim, uint32_t... rest_dims> struct transfer_bytes<T, dim, rest_dims...> {
    static constexpr uint32_t bytes = dim*transfer_bytes<T, rest_dims...>::bytes;
};

}
}

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#include "dsmem.cuh"
#endif