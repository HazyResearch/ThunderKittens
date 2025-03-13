/**
 * @file
 * @brief An aggregate header of all gang (multi-gpu) operations defined by ThunderKittens
 */

#pragma once

#include <array>
#include <stdexcept>
#include "../../types/device/hood.cuh"

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
template <int DEVICE_ID, int HOOD_SIZE>
__device__ static inline void sync(hood<HOOD_SIZE> hood, int sync_id) {
    #if defined(__CUDA_ARCH__)
        static_assert(__CUDA_ARCH__ >= 900, 
            "Using gang::sync() requires CUDA compute capability >= 9.0 (Hopper or newer)");
    #endif

    // Compile-time check if device is part of the gang using fold expression
    if constexpr (!((DEVICE_ID == GPUS) || ...)) return;

    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }

    size_t gang_addr = hood.get_address(sync_id);
    unsigned int *mc_addr = reinterpret_cast<unsigned int*>(
    hood.mc_ptrs[DEVICE_ID]) + gang_addr;
    unsigned int *uc_addr = reinterpret_cast<unsigned int*>(
    hood.uc_ptrs[DEVICE_ID]) + gang_addr;

    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" 
                : : "l"(mc_addr), "n"(1) : "memory");
    
    asm volatile ("fence.proxy.alias;" ::: "memory");
    
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ac(*uc_addr);
    while (GANG_SIZE > ac.load(cuda::memory_order_acquire));
}

__device__ static inline bool is_in_gang(int dev_idx) {
    return ((dev_idx == GPUS) || ...);
}

template <ducks::pgl::all PGL>
__device__ static inline void all_reduce_add(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::all_reduce_add(pgl, dev_idx);
}

template <ducks::pgl::all PGL>
__device__ static inline void all_reduce_min(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::all_reduce_min(pgl, dev_idx);
}

template <ducks::pgl::all PGL>
__device__ static inline void all_reduce_max(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::all_reduce_max(pgl, dev_idx);
}

template <ducks::pgl::all PGL>
__device__ static inline void atomic_add(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::atomic_add(pgl, dev_idx);
}

template <ducks::pgl::all PGL>
__device__ static inline void atomic_min(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::atomic_min(pgl, dev_idx);
}

template <ducks::pgl::all PGL>
__device__ static inline void atomic_max(PGL &pgl, int dev_idx) {
    if (!is_in_gang(dev_idx)) return;
    kittens::atomic_max(pgl, dev_idx);
}
};

} // namespace kittens