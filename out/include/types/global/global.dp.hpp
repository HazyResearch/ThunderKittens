/**
 * @file
 * @brief An aggregate header file for all the global types defined by ThunderKittens.
 */

#pragma once

#ifdef KITTENS_HOPPER
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tma.dp.hpp"
#endif
#include "util.dp.hpp"
#include "gl.dp.hpp"
#include "cgl.dp.hpp"
