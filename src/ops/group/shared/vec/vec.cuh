/**
 * @file
 * @brief An aggregate header for group operations on shared vectors.
 */

#include "conversions.cuh"
#include "maps.cuh"
// no group vector reductions as they would require additional shared memory and synchronization, and those side effects just aren't worth it.
// warp vector reductions should be plenty fast in 99.9% of situations.