/**
 * @file
 * @brief An aggregate header of warp memory operations on vectors, where a single warp loads or stores data on its own.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared_to_register.dp.hpp"
#include "global_to_register.dp.hpp"
#include "global_to_shared.dp.hpp"
#include "pgl_to_register.dp.hpp"
#include "pgl_to_shared.dp.hpp"

#ifdef KITTENS_HOPPER
#include "tma.dp.hpp"
#endif