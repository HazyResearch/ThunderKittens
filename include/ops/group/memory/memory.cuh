/**
 * @file
 * @brief An aggregate header of colaborative group memory movement operations
 */

#include "util/util.cuh"
#include "tile/tile.cuh"
#include "vec/vec.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
struct tma {
#include "util/tma.cuh"
#include "tile/tma.cuh"
#include "vec/tma.cuh"
struct cluster {
#include "util/tma_cluster.cuh"
#include "tile/tma_cluster.cuh"
#include "vec/tma_cluster.cuh"
};
};
#endif