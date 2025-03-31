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

    // Sync all threads in the block & make any previous memory ops visible
    // This must be done first because each block exits this function at different times
    __threadfence_system();
    __syncthreads();

    // All the code from here is simply asking every thread/block on all devices "did you arrive here yet?"
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        sync_point sp = sm.get_sync_point(sync_id, dev_id);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> uc0(sp.uc[0]);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_system> uc1(sp.uc[1]);
        cuda::atomic_ref<typename SyncManager::SYNC_SPACE_DTYPE, cuda::thread_scope_device> uc2(sp.uc[2]);

        // Whichever block reaches here first will become the leader block
        // We need to do it this way because there might be too many blocks to fit in the GPU at once
        typename SyncManager::SYNC_SPACE_DTYPE expected = 0; // atomic api requires pass by reference
        if (uc0.compare_exchange_strong(expected, 1, cuda::memory_order_acquire)) {

            // Sync the gang
            asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                        :: "l"(&sp.mc[1]), "n"(1) : "memory");
            asm volatile ("{fence.proxy.alias;}" ::: "memory");
            while (NUM_DEVICES > uc1.load(cuda::memory_order_acquire));
            
            // At this point:
            //   - All blocks within the same device are waiting for the leader block to notify
            //   - All devices' leader blocks have reached this point

            // Notify all non-leader blocks to proceed
            uc0++; // becomes 2

            // Wait for all non-leader blocks to exit
            size_t nblocks = gridDim.x * gridDim.y * gridDim.z;
            while (uc2.load(cuda::memory_order_acquire) < nblocks - 1);

            // All non-leader blocks went through. Now the leader block can clean up and proceed
            uc1.store(0, cuda::memory_order_release);
            uc2.store(0, cuda::memory_order_release);

            // This has to be stored last, as its acts as a mutex and the next sync() call might overlap
            uc0.store(0, cuda::memory_order_release);
        } else {
            // Wait for the leader block to finish sync'ing
            while (uc0.load(cuda::memory_order_acquire) != 2);
            uc2++;
        }
    }

    // Make all the threads wait for thread 0
    __syncthreads();
}

};

} // namespace kittens
