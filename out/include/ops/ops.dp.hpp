/**
 * @file
 * @brief A collection of all of the operations that ThunderKittens defines.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "warp/warp.dp.hpp"
#include "group/group.dp.hpp"
#include "gang/gang.dp.hpp"