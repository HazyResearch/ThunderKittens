#pragma once

/**
 * @file shared.cuh
 * @brief Header for shared memory operations utilizing vector types and conversions.
 *
 * This file includes the necessary operations for handling shared memory within CUDA kernels,
 * leveraging the functionality provided by vector operations and type conversions defined in
 * 'vec.cuh' and 'conversions.cuh'. It serves as a foundational component for optimized memory
 * access patterns in warp-level computations.
 */

#include "vec.cuh"
#include "conversions.cuh"
