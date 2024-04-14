#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/shared/shared.cuh"

namespace kittens {
namespace dsmem {

/**
 * @brief Distributes data from a source shared vector to a destination shared vector across different thread blocks.
 *
 * This function wraps the distribute function by automatically calculating the number of bytes to be transferred
 * based on the shared vector type and optional dimensions provided. It facilitates the distribution of data across
 * different clusters or thread blocks in a device.
 *
 * @tparam SV The shared vector type.
 * @tparam dims Variadic template parameter representing the dimensions of the array of shared vectors to be distributed.
 * @param[in,out] dst_ Reference to the destination shared vector.
 * @param[in,out] src_ Reference to the source shared vector.
 * @param[in] cluster_size The size of the cluster or the number of thread blocks involved in the distribution.
 * @param[in] dst_idx The index of the destination thread block within the cluster.
 * @param[in,out] bar Reference to a barrier used for synchronization across thread blocks.
 */
template<ducks::sv::all SV, uint32_t... dims>
__device__ static inline void distribute(SV &dst_, SV &src_, int cluster_size, int dst_idx, barrier& bar) {
    distribute(dst_, src_, cluster_size, dst_idx, transfer_bytes<SV, dims...>::bytes, bar); // wrap with auto calculated bytes
}

}
}