/**
 * @file
 * @brief An aggregate header of all gang (multi-gpu) operations defined by ThunderKittens
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <array>
#include <stdexcept>

namespace kittens {

/**
 * @brief Hood class manages multicast memory spaces across multiple GPUs for synchronization
 * @tparam HOOD_SIZE Number of GPUs that will be part of the hood
 */
template <int HOOD_SIZE>
class hood {
public:
/**
 * @brief Construct a new hood object
 * @param device_ids Array of device IDs to include in this hood
 * @throws std::runtime_error if CUDA initialization fails
 */
explicit hood(const int* device_ids) : num_devices(HOOD_SIZE) {
    cuInit(0);

    for (int i = 0; i < num_devices; ++i) {
        device_ids_[i] = device_ids[i];
    }

    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = num_devices;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
    mc_prop.flags = 0; // SBZ

    cuMulticastGetGranularity(&mc_granularity_, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
    mc_prop.size = mc_granularity_;

    cuMulticastCreate(&mc_handle_, &mc_prop);

    for (int i = 0; i < num_devices; ++i) {
        CUdevice dev;
        cuDeviceGet(&dev, device_ids_[i]);
        cuMulticastAddDevice(mc_handle_, dev);
    }

    initialize_memory();
}

/**
 * @brief Destroy the hood object and clean up resources
 */
~hood() {
    cleanup();
}

/**
 * @brief Get the memory address offset for a synchronization point
 * @param sync_id Unique identifier for the sync point
 * @return Address offset for the sync point
 */
size_t get_address(int sync_id) {
    if (sync_id_to_va_offset_.count(sync_id) == 0) {
        sync_id_to_va_offset_[sync_id] = cur_va_offset_;
        cur_va_offset_ += sizeof(unsigned int);
        
        // Check if we've exceeded the granularity
        if (cur_va_offset_ > mc_granularity_) {
            throw std::runtime_error("Exceeded multicast memory granularity");
        }
    } 
    return sync_id_to_va_offset_[sync_id];
}

private:
void initialize_memory() {
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(device_ids_[i]);

        CUmemAllocationProp memProp = {};
        memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        memProp.location.id = device_ids_[i];
        memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        size_t handle_granularity = 0;
        cuMemGetAllocationGranularity(
            &handle_granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        );

        cuMemCreate(&uc_handles_[i], handle_granularity, &memProp, 0);
        cuMulticastBindMem(mc_handle_, 0, uc_handles_[i], 0, mc_granularity_, 0);

        cuMemAddressReserve(&mc_ptrs_[i], mc_granularity_, 0, 0, 0);
        cuMemAddressReserve(&uc_ptrs_[i], mc_granularity_, 0, 0, 0);

        cuMemMap(mc_ptrs_[i], mc_granularity_, 0, mc_handle_, 0);
        cuMemMap(uc_ptrs_[i], mc_granularity_, 0, uc_handles_[i], 0);

        CUmemAccessDesc desc = {}; 
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc.location.id = device_ids_[i];
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        cuMemSetAccess(mc_ptrs_[i], mc_granularity_, &desc, 1);
        cuMemSetAccess(uc_ptrs_[i], mc_granularity_, &desc, 1);

        cudaMemset(reinterpret_cast<void*>(mc_ptrs_[i]), 0, mc_granularity_);
        cudaMemset(reinterpret_cast<void*>(uc_ptrs_[i]), 0, mc_granularity_);
    }
}

void cleanup() {
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(device_ids_[i]);
        cuMemUnmap(mc_ptrs_[i], mc_granularity_);
        cuMemUnmap(uc_ptrs_[i], mc_granularity_);
        cuMemAddressFree(mc_ptrs_[i], mc_granularity_);
        cuMemAddressFree(uc_ptrs_[i], mc_granularity_);
        cuMemRelease(uc_handles_[i]);
    }
    cuMulticastDestroy(mc_handle_);
}

public:
    const int num_devices;

private:
    int device_ids_[HOOD_SIZE];
    CUdeviceptr mc_ptrs_[HOOD_SIZE] = {};
    CUdeviceptr uc_ptrs_[HOOD_SIZE] = {};
    
    size_t mc_granularity_ = 0;
    size_t cur_va_offset_ = 0;
    
    CUmemGenericAllocationHandle mc_handle_ = {};
    CUmemGenericAllocationHandle uc_handles_[HOOD_SIZE] = {};
    
    std::unordered_map<int, size_t> sync_id_to_va_offset_;
};


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
 * @param hood_obj A kittens::hood object
 * @param sync_id Identifier for this synchronization point
 * @param device_id Current device ID
 */
__device__ static inline void sync(hood<GANG_SIZE> hood, int sync_id, int device_id) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }

    constexpr bool is_gang_member = ((GPUS == device_id) || ...);
    if (!is_gang_member) return;

    size_t gang_addr = gn.get_address(sync_id);
    unsigned int *mc_addr = reinterpret_cast<unsigned int*>(gn.mc_ptrs[device_id]) + gang_addr;
    unsigned int *uc_addr = reinterpret_cast<unsigned int*>(gn.uc_ptrs[device_id]) + gang_addr;

    #if __CUDA_ARCH__ >= 900
        // Increment arrival counter with release semantics
        asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" 
                    : : "l"(mc_addr), "n"(1) : "memory");
        
        // Fence between MC and UC access
        asm volatile ("fence.proxy.alias;" ::: "memory");
        
        // Spin-wait until all GPUs have arrived
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ac(*uc_addr);
        while (GANG_SIZE > ac.load(cuda::memory_order_acquire));
    #else
        printf("This barrier requires CUDA compute capability >= 9.0 (Hopper or newer)\n");
    #endif
}

};

} // namespace kittens
