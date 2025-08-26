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

struct everyone {
    /**
     * @brief Synchronizes all threads in all blocks in all GPUs in the gang.
     * 
     * Caution: if not all thread blocks are active on a GPU device, a deadlock will occur.
     */
    template <ducks::sync_manager::all SyncManager>
    __device__ static inline void sync(const SyncManager &sm, const int dev_idx) {
        #if defined(__CUDA_ARCH__)
            static_assert(__CUDA_ARCH__ >= 900, 
                "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
        #endif
        static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
            "Number of devices in the gang cannot be greater than that in the sync manager");

        // TODO: support a subset of devices
        if (dev_idx >= NUM_DEVICES) return;

        // Must do threadfence & syncthreads here, as there is no guarantee on the order of 
        // function exit after the synchronization.
        __threadfence_system();
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            sync_point sp = sm.get_all_sync_point(dev_idx);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> sys_uc(*sp.sys_uc);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> dev_uc(*sp.dev_uc);
    
            if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y == 0) { // Block 0 is the leader block
                size_t nblocks = gridDim.x * gridDim.y * gridDim.z;
                while (dev_uc.load(cuda::memory_order_acquire) < nblocks - 1); // Wait for all non-leader blocks to check in
    
                // At this point, all threads across all blocks on the current device have now reached 
                // gang::grid_sync() and are waiting.
    
                // Now, sync all leader blocks across all devices
                asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                    :: "l"(sp.sys_mc), "n"(1) : "memory");
                asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
                while (sys_uc.load(cuda::memory_order_acquire) < NUM_DEVICES);
    
                // At this point, all threads across all blocks on all devices have now reached
                // gang::grid_sync() and are waiting.
    
                // Release all blocks
                sys_uc.store(0, cuda::memory_order_release); // Do this before releasing non-leader blocks
                dev_uc.store(0, cuda::memory_order_release); // Release non-leader blocks
            } else {
                dev_uc++; // "check-in"
                while (dev_uc.load(cuda::memory_order_acquire) > 0);
            }
        }

        // Must block all threads until thread 0 completes the sync
        __syncthreads();
    }
};

struct blockwise {
    /**
     * @brief Synchronizes the blocks with the same block index running in all GPUs in the gang.
     * 
     * Ex. block idx N on all GPUs will be synchronized with respect to each other.
     *     But block idx N on one device will not be synchronized with block idx N + 1 on any device.
     * 
     * Caution: if the total number of active thread blocks on a GPU device exceeds approximately 
     *          twice the hardware limit, a deadlock may occur.
     */
    template <ducks::sync_manager::all SyncManager>
    __device__ static inline void sync(const SyncManager &sm, const int dev_idx) {
        #if defined(__CUDA_ARCH__)
            static_assert(__CUDA_ARCH__ >= 900, 
                "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
        #endif
        static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
            "Number of devices in the gang cannot be greater than that in the sync manager");
        static_assert(SyncManager::max_blocks > 0, "gang::blockwise::sync() requires a sync manager with MAX_BLOCKS > 0");

        if (dev_idx >= NUM_DEVICES) return;
    
        int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        if (block_idx >= SyncManager::max_blocks) return; // ignore blocks that are not in the sync_manager

        __threadfence_system();
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            sync_point sp = sm.get_blockwise_sync_point(dev_idx, block_idx);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> sys_uc(*sp.sys_uc);

            // Block-level gang sync
            asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                :: "l"(sp.sys_mc), "n"(1) : "memory");
            asm volatile ("{fence.proxy.alias;}" ::: "memory"); // nvidia says this is needed
            while (sys_uc.load(cuda::memory_order_acquire) < NUM_DEVICES);
    
            // All devices synced. Now clean up
            sys_uc.store(0, cuda::memory_order_release);

            // Must wait for all devices to finish cleaning up
            uint32_t val = 9999;
            do {
                asm volatile ("{multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];}": 
                    "=r"(val) : "l"(sp.sys_mc) : "memory");
            } while (val > 0);
        }

        __syncthreads();
    }
};

struct blockgroup {
    /**
     * @brief Synchronizes the given number of blocks running in all GPUs in the gang.
     * 
     * Ex. to synchronize blocks 2, 3, 4 on dev 0, and blocks 111, 112 on dev 1, call this 
     *     function on those blocks with the same sync_id and expected_arrivals = 5
     * 
     * Caution: if you use the same sync ID for multiple syncs across different set of blocks, 
     *          the behavior is undefined. A deadlock may occur for incorrect usage.
     */
    template <ducks::sync_manager::all SyncManager>
    __device__ static inline void sync(const SyncManager &sm, const int dev_idx, const int sync_id, const int expected_arrivals) {
        #if defined(__CUDA_ARCH__)
            static_assert(__CUDA_ARCH__ >= 900, 
                "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
        #endif
        static_assert(NUM_DEVICES <= SyncManager::SYNC_SPACE_T::num_devices, 
            "Number of devices in the gang cannot be greater than that in the sync manager");
        static_assert(SyncManager::max_sync_points > 0, "gang::blockgroup::sync() requires a sync manager with MAX_SYNC_POINTS > 0");

        if (dev_idx >= NUM_DEVICES) return;

        __threadfence_system();
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            sync_point sp = sm.get_blockgroup_sync_point(dev_idx, sync_id);
            cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> sys_uc(*sp.sys_uc);

            typename SyncManager::SYNC_SPACE_DTYPE expected = 0; // atomic api requires pass by reference
            if (sys_uc.compare_exchange_strong(expected, 1, cuda::memory_order_acquire)) {
                // Wait for the number of block arrivals across all devices to reach the expected number
                uint32_t num_arrivals = 0;
                do {
                    asm volatile ("{multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];}": 
                        "=r"(num_arrivals) : "l"(sp.sys_mc) : "memory");
                } while (num_arrivals < expected_arrivals);

                // Clean up & release all blocks
                sys_uc.store(0, cuda::memory_order_release); 
            } else { // other block took the leadership and initiated the sync
                ++sys_uc; // check-in
                while (sys_uc.load(cuda::memory_order_acquire) != 0); // Wait for the leader to finish the sync
            }
        }

        __syncthreads();
    }
};

}; // struct gang

} // namespace kittens
