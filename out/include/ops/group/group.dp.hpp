/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by ThunderKittens
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../../common/common.dp.hpp"
#include "../../types/types.dp.hpp"
#include "../warp/warp.dp.hpp" // several group memory ops rely on underlying warp-scope ops

// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP static_assert(N_WARPS==4, "PTX warpgroup (N_WARPS=4) function called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within the group struct, so we declare them in advance.
#ifdef KITTENS_HOPPER
#include "wgmma/base/base.dp.hpp"
#endif

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct group {
static constexpr int GROUP_WARPS = N_WARPS; // This alias produces nice parallelism.
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
static inline int laneid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
               GROUP_THREADS;
}
static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
static inline int groupid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) /
               GROUP_THREADS;
}

static inline void sync(int id) {
    /*
    DPCT1053:280: Migration of device assembly code is not supported.
    */
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(GROUP_THREADS));
}
static inline void arrive(int id) {
    /*
    DPCT1053:281: Migration of device assembly code is not supported.
    */
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(GROUP_THREADS));
}

#include "memory/memory.dp.hpp"
#include "shared/shared.dp.hpp"
#include "register/register.dp.hpp"

#ifdef KITTENS_HOPPER
#include "wgmma/wgmma.dp.hpp"

template<int n_reg> static inline void increase_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    /*
    DPCT1053:302: Migration of device assembly code is not supported.
    */
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(n_reg));
}
template<int n_reg> static inline void decrease_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    /*
    DPCT1053:303: Migration of device assembly code is not supported.
    */
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(n_reg));
}
static inline void producer_registers() { decrease_registers<24>(); }
template<int NCWG> static inline void consumer_registers() { increase_registers<480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)>(); }

#endif

};

using warpgroup = group<4>; // special scope commonly used by SM_90 and later.

}