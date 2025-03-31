/**
 * @file
 * @brief An aggregate header of all gang (multi-gpu) operations defined by ThunderKittens
 */

#pragma once

#include <array>
#include <stdexcept>
#include "../../types/device/sync_manager.cuh"

namespace kittens {

/**
 * @brief Gang template represents a collection of GPUs working together
 */
template <int NUM_DEVICES>
struct gang {

/**
 * @brief Synchronize all GPUs in a gang at a specific sync point. 
 *        This is a block-level sync (i.e., only thread blocks with
 *        the same block ID across all GPUs will be synchronized).
 *        Be warned that if the number of thread blocks is too large 
 *        (more than 2x the HW maximum), there might be a deadlock if
 *        two GPU devices schedule completely different set of blocks.
 */
template <ducks::sync_manager::all SyncManager>
__device__ static inline void sync(const SyncManager &sm, const int sync_id, const int dev_id) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif
    static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
        "Number of devices in the gang cannot be greater than that in the sync manager");

    // TODO: support a subset of devices
    if (dev_id >= NUM_DEVICES) return;

    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    if (block_idx >= SyncManager::max_blocks) return; // ignore blocks that are not in the sync_manager

    // It is important to set threadfence & syncthreads here, as there is no guarantee on the order of
    // operations on the same blocks on different devices after the multimem.red operation below.
    // If we do this at the end of this function, following multigpu operations may not observe data properly
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sync_point sp = sm.get_sync_point(sync_id, dev_id, block_idx);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> uc(*sp.uc);

        // Block-level gang sync
        asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
            :: "l"(sp.mc), "n"(1) : "memory");
        asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
        while (uc.load(cuda::memory_order_acquire) < NUM_DEVICES);

        // All devices synced. Now clean up and proceed
        uc.store(0, cuda::memory_order_release);
    }

    // Must block all threads until thread 0 completes the sync
    // Very certain that the two syncthreads are both needed
    __syncthreads();
}

};

} // namespace kittens
