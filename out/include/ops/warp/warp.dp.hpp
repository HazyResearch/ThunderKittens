/**
 * @file
 * @brief An aggregate header of all warp (worker) operations defined by ThunderKittens
 */

#pragma once

// no namespace wrapper needed here
// as warp is the default op scope!

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "register/register.dp.hpp"
#include "shared/shared.dp.hpp"
#include "memory/memory.dp.hpp"