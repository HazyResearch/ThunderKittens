/**
 * @file
 * @brief An aggregate header of warp operations on data in shared memory
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tile/tile.dp.hpp"
#include "vec/vec.dp.hpp"