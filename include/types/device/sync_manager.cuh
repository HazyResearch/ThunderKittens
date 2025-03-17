#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory> 
#include "detail/helpers.cuh"

namespace kittens {

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// TODO: for a given address, need to make sure address is reset back to zero
// or some other measure to ensure that the address can be reused
struct SyncSpace {
public: 
    SyncSpace(CUdeviceptr mc_ptr, CUdeviceptr uc_ptr, int size, int dev_id)
        : mc_ptr(mc_ptr), uc_ptr(uc_ptr), size(size), dev_id(dev_id) {}
    
    CUdeviceptr mc_ptr;
    CUdeviceptr uc_ptr;
    int dev_id;

    /**
     * @brief Get the memory address offset for a synchronization point
     * @param sync_id Unique identifier for the sync point
     * @return Address offset for the sync point
     */
    __device__ inline size_t get_address(int sync_id) const {
        assert(sync_id < size / sizeof(unsigned int));
        return sync_id * sizeof(unsigned int);
    }
private:
    int size;
};

/**
 * @brief The SyncManager class manages multicast memory spaces across 
 * multiple GPUs for synchronization. All "gangs" in TK are part of the same SyncManager.
 */
class SyncManager {
public:
    /**
     * @brief Construct a new SyncManager object
     * @param device_ids Array of device IDs to include in this SyncManager
     * @throws std::runtime_error if CUDA initialization fails
     */
    explicit SyncManager(int hood_size, const int* device_ids) : num_devices(hood_size) {
        mc_ptrs = new CUdeviceptr[hood_size];
        uc_ptrs = new CUdeviceptr[hood_size];
        device_ids_ = new int[hood_size];
        uc_handles_ = new CUmemGenericAllocationHandle[hood_size];
        
        // Store device IDs for later use
        for (int i = 0; i < hood_size; ++i) {
            device_ids_[i] = device_ids[i];
        }

        cuInit(0);

        CUmulticastObjectProp mcProp = {};
        mc_size_ = detail::init_mc_prop(&mcProp, hood_size);
     
        CUCHECK(cuMulticastCreate(&mc_handle_, &mcProp));

        // Add devices to the multicast object
        for (int dev_idx = 0; dev_idx < hood_size; ++dev_idx) {
            CUCHECK(cuMulticastAddDevice(mc_handle_, device_ids[dev_idx]));
        }
        
        for (int dev_idx = 0; dev_idx < hood_size; ++dev_idx) {
            CUDACHECK(cudaSetDevice(device_ids[dev_idx]));
            
            CUmemAllocationProp memProp = detail::create_mem_prop(device_ids[dev_idx]);

            size_t handle_granularity = 0;
            CUCHECK(cuMemGetAllocationGranularity(
                &handle_granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            ));

            CUCHECK(cuMemCreate(&uc_handles_[dev_idx], mc_size_, &memProp, 0));

            // Bind each unicast allocation to the multicast object at offset 0
            CUCHECK(cuMulticastBindMem(
                mc_handle_, 
                /*mcOffset*/ 0,
                uc_handles_[dev_idx],
                /*memOffset*/ 0, 
                mc_size_, 
                0
            ));
        }

        // Reserve address space and map memory for multicast and unicast accesses
        for (int dev_idx = 0; dev_idx < hood_size; ++dev_idx) {
            CUDACHECK(cudaSetDevice(device_ids[dev_idx]));

            CUCHECK(cuMemAddressReserve(&mc_ptrs[dev_idx], mc_size_, mc_size_, 0, 0));
            CUCHECK(cuMemAddressReserve(&uc_ptrs[dev_idx], mc_size_, mc_size_, 0, 0));

            CUCHECK(cuMemMap(mc_ptrs[dev_idx], mc_size_, 0, mc_handle_, 0));
            CUCHECK(cuMemMap(uc_ptrs[dev_idx], mc_size_, 0, uc_handles_[dev_idx], 0));

            CUmemAccessDesc desc = detail::create_mem_desc(device_ids[dev_idx]);
            CUCHECK(cuMemSetAccess(mc_ptrs[dev_idx], mc_size_, &desc, 1));
            CUCHECK(cuMemSetAccess(uc_ptrs[dev_idx], mc_size_, &desc, 1));
            CUDACHECK(cudaMemset(reinterpret_cast<void*>(mc_ptrs[dev_idx]), 0, mc_size_));
            CUDACHECK(cudaMemset(reinterpret_cast<void*>(uc_ptrs[dev_idx]), 0, mc_size_));
        }
    }

    /**
     * @brief Destroy the SyncManager object and clean up resources
     */
    ~SyncManager() {
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(device_ids_[i]);
            cuMemUnmap(mc_ptrs[i], mc_size_);
            cuMemUnmap(uc_ptrs[i], mc_size_);
            cuMemAddressFree(mc_ptrs[i], mc_size_);
            cuMemAddressFree(uc_ptrs[i], mc_size_);
            cuMemRelease(uc_handles_[i]);
        }
        cuMemRelease(mc_handle_);
    }

    SyncSpace get_sync_space(int dev_id) const {
        return SyncSpace(mc_ptrs[dev_id], uc_ptrs[dev_id], mc_size_, dev_id);
    }


    SyncManager(const SyncManager&) = delete;
    SyncManager& operator=(const SyncManager&) = delete;
    SyncManager(SyncManager&& other) noexcept;
    SyncManager& operator=(SyncManager&& other) noexcept;

public:
    const int num_devices;
    CUdeviceptr *mc_ptrs;
    CUdeviceptr *uc_ptrs;
    
private:
    int *device_ids_;
    
    size_t mc_size_ = 0;
    size_t cur_va_offset_ = 0;
    
    CUmemGenericAllocationHandle mc_handle_;
    CUmemGenericAllocationHandle *uc_handles_;
};

} // namespace kittens