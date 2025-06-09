#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file
 * @brief Various utilities for group memory operations.
 */

template<int N=0> static inline void load_async_wait(int bar_id) { // for completing (non-TMA) async loads
    /*
    DPCT1026:282: The call to "cp.async.wait_group %0;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
    */

    sync(bar_id);
}