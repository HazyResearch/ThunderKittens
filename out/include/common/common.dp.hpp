/**
 * @file
 * @brief A collection of common resources on which ThunderKittens depends.
 */
 

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "util.dp.hpp"
#include "base_types.dp.hpp"
#include "base_ops.dp.hpp"