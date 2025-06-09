/**
 * @file
 * @brief An aggregate header for warp operations on data stored in registers.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tile/tile.dp.hpp"
#include "vec/vec.dp.hpp"