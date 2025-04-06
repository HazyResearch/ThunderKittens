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
 * @brief Synchronizes all GPUs in a gang at a specific synchronization point.
 *
 * This is a block-level synchronization, meaning that only thread blocks
 * with the same block ID across all GPUs will be synchronized. 
 *
 * Caution: If the total number of active thread blocks on a GPU device 
 * exceeds approximately twice the hardware limit, a deadlock may occur.
 */
template <ducks::sync_manager::all SyncManager>
__device__ static inline void block_sync(const SyncManager &sm, const int sync_id, const int dev_idx) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::block_sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif
    static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
        "Number of devices in the gang cannot be greater than that in the sync manager");
    static_assert(!SyncManager::is_grid_sync, "gang::block_sync() cannot be used with grid-level sync manager");

    // TODO: support a subset of devices
    if (dev_idx >= NUM_DEVICES) return;

    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    if (block_idx >= SyncManager::max_blocks) return; // ignore blocks that are not in the sync_manager

    // It is important to set threadfence & syncthreads here, as there is no guarantee on the order of
    // operations on the same blocks on different devices after the multimem.red operation below.
    // If we do this at the end of this function, following multigpu operations may not observe data properly
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sync_point sp = sm.get_sync_point(sync_id, dev_idx, block_idx);
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
    __syncthreads();
}

/**
 * @brief Synchronizes all GPUs in a gang at a specific synchronization point.
 *
 * This is a grid-level synchronization, meaning that all thread blocks across
 * all GPUs will be synchronized.
 *
 * Caution: If not all thread blocks are active on a GPU device, a deadlock will occur.
 */
template <ducks::sync_manager::all SyncManager>
__device__ static inline void grid_sync(const SyncManager &sm, const int sync_id, const int dev_idx) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::grid_sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif

    if (dev_idx >= NUM_DEVICES) return;

    __threadfence_system();
    __syncthreads();

    // All the code from here is simply asking every thread/block on all devices "did you arrive here yet?"
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sync_point sp = sm.get_sync_point(sync_id, dev_idx);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> uc0(sp.uc[0]);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> uc1(sp.uc[1]);

        int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        if (block_idx == 0) { // Block 0 is the leader block
            size_t nblocks = gridDim.x * gridDim.y * gridDim.z;
            while (uc0.load(cuda::memory_order_acquire) < nblocks - 1); // Wait for all non-leader blocks to check in

            // At this point, all threads across all blocks on the current device have now reached 
            // gang::grid_sync() and are waiting.

            // Now, sync all leader blocks across all devices
            asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                :: "l"(&sp.mc[1]), "n"(1) : "memory");
            asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
            while (uc1.load(cuda::memory_order_acquire) < NUM_DEVICES);

            // At this point, all threads across all blocks on all devices have now reached
            // gang::grid_sync() and are waiting.

            // Release all blocks
            uc1.store(0, cuda::memory_order_release); // Do this before releasing non-leader blocks
            uc0.store(0, cuda::memory_order_release); // Release non-leader blocks
        } else {
            uc0++; // "check-in"
            while (uc0.load(cuda::memory_order_acquire) > 0);
        }
    }

    // Make all threads wait for thread 0
    __syncthreads();
}

};

} // namespace kittens
