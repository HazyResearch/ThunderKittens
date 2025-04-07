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
 * If LEVEL is sync_level::BLOCK, performs a block-level synchronization, 
 * meaning that only thread blocks with the same block ID across all GPUs 
 * will be synchronized. If LEVEL is sync_level::GRID, performs a grid-level
 * synchronization: all thread blocks across all GPUs will be synchronized.
 * 
 * Caution: There are deadlock conditions.
 * - For BLOCK level, if the total number of active thread blocks on a GPU device 
 * exceeds approximately twice the hardware limit, a deadlock may occur.
 * - For GRID level, if not all thread blocks are active on a GPU device, a deadlock will occur.
 */
template <sync_level LEVEL, ducks::sync_manager::all SyncManager>
__device__ static inline void sync(const SyncManager &sm, const int sync_id, const int dev_idx) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif
    static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
        "Number of devices in the gang cannot be greater than that in the sync manager");
    static_assert(LEVEL == SyncManager::level, "Gang sync level must match the sync manager sync level");

    // TODO: support a subset of devices
    if (dev_idx >= NUM_DEVICES) return;

    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    if constexpr (LEVEL == sync_level::BLOCK) {
        if (block_idx >= SyncManager::sync_point_size) return; // ignore blocks that are not in the sync_manager
    }

    // Must do threadfence & syncthreads here, as there is no guarantee on the order of 
    // function exit after the synchronization.
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        if constexpr (LEVEL == sync_level::GRID) {
            sync_point sp = sm.get_sync_point(sync_id, dev_idx);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> gang_uc(*sp.gang_uc);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> grid_uc(*sp.grid_uc);
    
            if (block_idx == 0) { // Block 0 is the leader block
                size_t nblocks = gridDim.x * gridDim.y * gridDim.z;
                while (grid_uc.load(cuda::memory_order_acquire) < nblocks - 1); // Wait for all non-leader blocks to check in
    
                // At this point, all threads across all blocks on the current device have now reached 
                // gang::grid_sync() and are waiting.
    
                // Now, sync all leader blocks across all devices
                asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                    :: "l"(sp.gang_mc), "n"(1) : "memory");
                asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
                while (gang_uc.load(cuda::memory_order_acquire) < NUM_DEVICES);
    
                // At this point, all threads across all blocks on all devices have now reached
                // gang::grid_sync() and are waiting.
    
                // Release all blocks
                gang_uc.store(0, cuda::memory_order_release); // Do this before releasing non-leader blocks
                grid_uc.store(0, cuda::memory_order_release); // Release non-leader blocks
            } else {
                grid_uc++; // "check-in"
                while (grid_uc.load(cuda::memory_order_acquire) > 0);
            }
        } else if constexpr (LEVEL == sync_level::BLOCK) {
            sync_point sp = sm.get_sync_point(sync_id, dev_idx, block_idx);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> gang_uc(*sp.gang_uc);
    
            // Block-level gang sync
            asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                :: "l"(sp.gang_mc), "n"(1) : "memory");
            asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
            while (gang_uc.load(cuda::memory_order_acquire) < NUM_DEVICES);
    
            // All devices synced. Now clean up and proceed
            gang_uc.store(0, cuda::memory_order_release);
        }
    }

    // Must block all threads until thread 0 completes the sync
    __syncthreads();
}

template <sync_level LEVEL, ducks::sync_manager::all SyncManager>
__device__ static inline void new_sync(const SyncManager &sm, const int sync_id, const int expected_arrivals, const int dev_idx) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif
    static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
        "Number of devices in the gang cannot be greater than that in the sync manager");
    static_assert(LEVEL == SyncManager::level, "Gang sync level must match the sync manager sync level");

    // TODO: support a subset of devices
    if (dev_idx >= NUM_DEVICES) return;

    // Must do threadfence & syncthreads here, as there is no guarantee on the order of 
    // function exit after the synchronization.
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sync_point sp = sm.get_sync_point(sync_id, dev_idx);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> gang_uc(*sp.gang_uc);

        typename SyncManager::SYNC_SPACE_DTYPE expected = 0; // atomic api requires pass by reference
        if (gang_uc.compare_exchange_strong(expected, 1, cuda::memory_order_acquire)) {

            // Wait for the number of block arrivals across all devices to reach the expected number
            uint32_t num_arrivals = 0;
            do {
                asm volatile ("{multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];}": 
                    "=r"(num_arrivals) : "l"(sp.gang_mc) : "memory");
            } while (num_arrivals < expected_arrivals);

            // Clean up & release all blocks
            gang_uc.store(0, cuda::memory_order_release); 

        } else { // other block took the leadership and initiated the sync
            ++gang_uc; // check-in
            while (gang_uc.load(cuda::memory_order_acquire) != 0); // Wait for the leader to finish the sync
        }
    }

    // Must block all threads until thread 0 completes the sync
    __syncthreads();
}

};

} // namespace kittens
