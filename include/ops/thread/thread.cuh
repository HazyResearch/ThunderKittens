/**
 * @file
 * @brief An aggregate header of all single-threaded operations defined by ThunderKittens
 */

#pragma once

// no namespace wrapper needed here
#include "util.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "multimem.cuh"
#include "tma_util.cuh"
#include "tma_vec.cuh"
#include "tma_tile.cuh"
#endif

#if defined(KITTENS_BLACKWELL)
#include "tcgen05_util.cuh"
#include "tcgen05_mma.cuh"
#endif
