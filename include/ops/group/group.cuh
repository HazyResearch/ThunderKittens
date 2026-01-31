/**
 * @file
 * @brief An aggregate header of all group (one or more warps) operations defined by ThunderKittens
 */

#pragma once

#include <cuda/pipeline>

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../thread/thread.cuh" // several group memory ops rely on underlying single-thread-scope ops

#define KITTENS_CHECK_WARP static_assert(GROUP_WARPS==1, "Warp (GROUP_WARPS=1) function called from a non-warp group.");
// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP static_assert(GROUP_WARPS==4, "Warpgroup (GROUP_WARPS=4) function called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within the group struct, so we declare them in advance.
#if defined(KITTENS_HOPPER)
#include "mma/base/base.cuh"
#endif

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int _GROUP_WARPS>
struct group {
static constexpr int GROUP_WARPS = _GROUP_WARPS; // This alias produces nice parallelism.
static constexpr int GROUP_THREADS = GROUP_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}
template<uint32_t MASK=0xFFFFFFFF> __device__ static inline void sync() {
    static_assert(GROUP_WARPS==1, "barrier-less sync() can only be called by a single warp!");
    asm volatile("bar.warp.sync %0;\n" :: "n"(MASK));
}
__device__ static inline void arrive(int id) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

#include "memory/memory.cuh"
#include "shared/shared.cuh"
#include "register/register.cuh"
#include "mma/mma.cuh"
#include "util/util.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

template<int n_reg> __device__ static inline void increase_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
template<int n_reg> __device__ static inline void decrease_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
__device__ static inline void producer_registers() { decrease_registers<24>(); }
template<int NCWG> __device__ static inline void consumer_registers() { increase_registers<480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)>(); }

// ---- TMA operations ----
// These must be included here because
//   1. We want parallel scope with single-thread ops (i.e., tma:: and tma::cluster)
//   1. We can't use namespaces as this is under struct group
//   2. Struct can't be declared in multiple places
struct tma {
#include "memory/tile/tma.cuh"
#include "memory/vec/tma.cuh"
#include "util/tma.cuh"
struct cluster {
#include "memory/tile/tma_cluster.cuh"
#include "memory/vec/tma_cluster.cuh"
#include "util/tma_cluster.cuh"
};
};

#endif

};

namespace everyone {

// Block-level synchronization
__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0;\n" :: "r"(id));
}

// Cluster-level synchronization functions
namespace tma {
namespace cluster {
__device__ static inline void arrive() { // All threads in the cluster must call this
    asm volatile ("barrier.cluster.arrive.release;\n");
}
__device__ static inline void arrive_aligned() { // All threads in the cluster must call this
    asm volatile ("barrier.cluster.arrive.release.aligned;\n");
}
__device__ static inline void wait() {
    asm volatile ("barrier.cluster.wait.acquire;\n");
}
__device__ static inline void wait_aligned() {
    asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
}
__device__ static inline void sync() {
    arrive_aligned();
    wait_aligned();
}
}
}

};

using warp = group<1>;      // scope used by most pre-Hopper GPUs and most register operations.
using warpgroup = group<4>; // special scope used by Hopper.

}
