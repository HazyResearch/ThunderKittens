/**
 * @file
 * @brief An aggregate header for warp operations on register vectors.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"
#include "maps.dp.hpp"
#include "reductions.dp.hpp"