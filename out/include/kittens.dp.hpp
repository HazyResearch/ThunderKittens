/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common/common.dp.hpp"
#include "types/types.dp.hpp"
#include "ops/ops.dp.hpp"
#include "pyutils/util.dp.hpp"
// #include "pyutils/pyutils.cuh" // for simple binding without including torch