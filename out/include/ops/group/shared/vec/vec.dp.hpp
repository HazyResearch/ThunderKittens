/**
 * @file
 * @brief An aggregate header for group operations on shared vectors.
 */

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"
#include "maps.dp.hpp"
// no group vector reductions as they would require additional shared memory and synchronization, and those side effects just aren't worth it.
// warp vector reductions should be plenty fast in 99.9% of situations.