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
 * @tparam GPUS Compile-time list of GPU device IDs in the gang
 */
template <int... GPUS>
struct gang {
static constexpr int GANG_SIZE = sizeof...(GPUS);
static constexpr std::array<int, sizeof...(GPUS)> gpu_ids{GPUS...};

/**
 * @brief Synchronize all GPUs in a gang at a specific sync point
 * @tparam DEVICE_ID The device ID of the calling GPU
 * @param hood_obj A kittens::hood object
 * @param sync_id Identifier for this synchronization point
 */
// template <int HOOD_SIZE>
__device__ static inline void sync(SyncSpace s, int sync_id = 0) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif

    if (!is_in_gang(s.dev_id)) return;

    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }

    unsigned int *mc_addr = reinterpret_cast<unsigned int*>(
        s.mc_ptr) + s.get_address(sync_id);
    unsigned int *uc_addr = reinterpret_cast<unsigned int*>(
        s.uc_ptr) + s.get_address(sync_id);

    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" 
                  :: "l"(mc_addr), "n"(1) : "memory");
    
    asm volatile ("fence.proxy.alias;" ::: "memory");

    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ac(*uc_addr);
    while (GANG_SIZE > ac.load(cuda::memory_order_acquire));
}

__device__ static inline bool is_in_gang(int dev_idx) {
    return ((dev_idx == GPUS) || ...);
}

// template <typename PGL_OBJ>
// __device__ static inline void reduce_add(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::all_reduce_add(pgl);
// }

// template <typename PGL_OBJ>
// __device__ static inline void reduce_min(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::all_reduce_min(pgl);
// }

// template <typename PGL_OBJ>
// __device__ static inline void reduce_max(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::all_reduce_max(pgl);
// }

// template <typename PGL_OBJ>
// __device__ static inline void atomic_add(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::atomic_add(pgl);
// }

// template <typename PGL_OBJ>
// __device__ static inline void atomic_min(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::atomic_min(pgl);
// }

// template <typename PGL_OBJ>
// __device__ static inline void atomic_max(PGL_OBJ pgl) {
//     if (!is_in_gang(pgl.dev_id)) return;
//     kittens::atomic_max(pgl);
// }
};

} // namespace kittens