/**
 * @file
 * @brief An aggregate header of group memory operations on tiles.
 */

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared_to_register.dp.hpp"
#include "global_to_register.dp.hpp"
#include "global_to_shared.dp.hpp"
#include "pgl_to_shared.dp.hpp"
#include "pgl_to_register.dp.hpp"

#include "complex/complex_shared_to_register.dp.hpp"
#include "complex/complex_global_to_register.dp.hpp"
#include "complex/complex_global_to_shared.dp.hpp"
