#pragma once

/**
 * @file register.cuh
 * @brief Aggregates includes for register-level operations within CUDA kernels.
 *
 * This file includes a set of headers that provide functionality for register-level
 * operations used in warp-level computations. It serves as a convenience header to
 * include common operations such as conversions, vector operations, mappings,
 * reductions, and matrix multiply-accumulate operations at the register level.
 */

#include "conversions.cuh"
#include "vec.cuh"
#include "maps.cuh"
#include "reductions.cuh"
#include "mma.cuh"
