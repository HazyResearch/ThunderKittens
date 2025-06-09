/**
 * @file
 * @brief A collection of all of ThunderKittens prototypes, that can be filled in to easily build full kernels.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../include/kittens.dp.hpp"

#include "common/common.dp.hpp"
#include "lcf/lcf.dp.hpp"
#include "lcsf/lcsf.dp.hpp"
#include "interpreter/interpreter.dp.hpp"