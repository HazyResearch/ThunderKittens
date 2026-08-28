/**
 * @file
 * @brief An aggregate header for all group-scope MMA operations.
 */

// All compilation targets can use the warp-scope MMA operations.
#include "warp.cuh"

// Hopper has its own warpgroup-scope MMA operations.
#if defined(KITTENS_SM90)
#include "warpgroup.cuh"
#elif defined(KITTENS_SM80) || defined(KITTENS_SM120)
#include "warpgroup_pseudo.cuh"
#endif

// Blackwell has its own MMA operations (Tensor Core Generation 5).
#ifdef KITTENS_SM10X
#include "tcgen05.cuh"
#endif
