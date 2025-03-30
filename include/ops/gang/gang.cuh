/**
 * @file
 * @brief An aggregate header of all gang (multi-gpu) operations defined by ThunderKittens
 */

#pragma once

#include <array>
#include <stdexcept>
#include "../../types/device/sync_manager.cuh"

namespace kittens {

// TODO: for a given address, need to make sure address is reset back to zero
// or some other measure to ensure that the address can be reused

/**
 * @brief Gang template represents a collection of GPUs working together
 */
template <int NUM_DEVICES>
struct gang {

/**
 * @brief Synchronize all GPUs in a gang at a specific sync point
 */
template <ducks::sync_manager::all SyncManager>
__device__ static inline void sync(const SyncManager &sm, const int sync_id, const int dev_id) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif
    
    // TODO: support a subset of devices
    if (dev_id >= NUM_DEVICES) return;
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    
    sync_point sp = sm.get_sync_point(sync_id, dev_id);

    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" 
                  :: "l"(sp.mc), "n"(1) : "memory");
    asm volatile ("fence.proxy.alias;" ::: "memory");
    cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> ac(*sp.uc);
    while (NUM_DEVICES > ac.load(cuda::memory_order_acquire));
}

};

} // namespace kittens