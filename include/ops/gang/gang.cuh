/**
 * @file
 * @brief An aggregate header of all gang (multi-gpu) operations defined by ThunderKittens
 */

 #pragma once

 #include <array>
 #include <stdexcept>
 #include "../../types/device/hood.cuh"
 
 namespace kittens {
 
template<int DeviceID, int... GangIDs>
struct is_device_in_gang;

template<int DeviceID>
struct is_device_in_gang<DeviceID> : std::false_type {};

template<int DeviceID, int CurrentGangID, int... RemainingGangIDs>
struct is_device_in_gang<DeviceID, CurrentGangID, RemainingGangIDs...> : 
    std::conditional_t<DeviceID == CurrentGangID, std::true_type, 
                       is_device_in_gang<DeviceID, RemainingGangIDs...>> {};

template<int DeviceID, int... GangIDs>
inline constexpr bool is_device_in_gang_v = is_device_in_gang<DeviceID, GangIDs...>::value;
 
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
 
         if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
             blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
             return;
         }
 
         // Compile-time check if device is part of the gang
         if constexpr (!is_device_in_gang_v<DEVICE_ID, GPUS...>) return;
 
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
 };
 
 } // namespace kittens