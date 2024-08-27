#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/shared/shared.cuh"
#include "../util/util.cuh"

namespace kittens {
namespace dsmem {

/**
 * @brief Distributes data from a source shared tile to a destination shared tile across different thread blocks.
 *
 * This function wraps the distribute function by automatically calculating the number of bytes to be transferred
 * based on the shared tile type and optional dimensions provided. It facilitates the distribution of data across
 * different clusters or thread blocks in a device.
 *
 * @tparam ST The shared tile type.
 * @tparam dims Variadic template parameter representing the dimensions of the array of shared tiles to be distributed.
 * @param[in,out] dst_ Reference to the destination shared tile.
 * @param[in,out] src_ Reference to the source shared tile.
 * @param[in] cluster_size The size of the cluster or the number of thread blocks involved in the distribution.
 * @param[in] dst_idx The index of the destination thread block within the cluster.
 * @param[in,out] bar Reference to a barrier used for synchronization across thread blocks.
 */
template<ducks::st::all ST, uint32_t... dims>
__device__ static inline void distribute(ST &dst_, ST &src_, int cluster_size, int dst_idx, barrier& bar) {
    distribute(dst_, src_, cluster_size, dst_idx, kittens::size_bytes<ST, dims...>, bar); // wrap with auto calculated bytes
}

}
}