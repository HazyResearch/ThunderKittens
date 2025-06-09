/**
 * @file
 * @brief An aggregate header of warp memory operations, where a single warp loads or stores data on its own.
 */

#pragma once

// #include "util/util.cuh"
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tile/tile.dp.hpp"
#include "vec/vec.dp.hpp"