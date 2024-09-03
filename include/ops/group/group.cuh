/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by ThunderKittens
 */

#pragma once

#include <cuda/pipeline>

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../warp/warp.cuh" // several group memory ops rely on underlying warp-scope ops

// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP static_assert(N_WARPS==4, "PTX warpgroup (N_WARPS=4) function called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within the group struct, so we declare them in advance.
#ifdef KITTENS_HOPPER
#include "wgmma/base/base.cuh"
#endif

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct group {
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

// TODO: redo these with good default barrier index choices
__device__ static inline void sync() { // warning: this can create trouble if multiple groups of different sizes are using it at the same time.
    asm volatile("bar.sync %0, %1;\n" :: "r"(groupid() + 4), "n"(GROUP_THREADS)); // +4 here is meant to avoid conflicts with bar.sync used by __syncthreads(), a common special case of the above concern.
}
__device__ static inline void sync(int id) { // backup: specify the barrier ID manually
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

#include "memory/memory.cuh"
#include "shared/shared.cuh"

#ifdef KITTENS_HOPPER
#include "wgmma/wgmma.cuh"

template<int n_reg> __device__ static inline void increase_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
template<int n_reg> __device__ static inline void decrease_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}

#endif

};

}