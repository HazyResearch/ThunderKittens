#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include "detail/helpers.cuh"

namespace kittens {

/**
 * @brief The hood (neighborhood) class manages multicast memory spaces across 
 * multiple GPUs for synchronization. All "gangs" in TK are part of the same hood.
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

    CUmulticastObjectProp mc_prop; 
    mc_size_ = detail::init_mc_prop(&mc_prop, num_devices);

    cuMulticastCreate(&mc_handle_, &mc_prop);
    for (int i = 0; i < num_devices; ++i) {
        CUdevice dev;
        cuDeviceGet(&dev, device_ids_[i]);
        cuMulticastAddDevice(mc_handle_, dev);
    }

    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(device_ids_[i]);
        cuMemAlloc((CUdeviceptr *)&uc_handles_[i], mc_size_);
        cuMulticastBindAddr(mc_handle_, 0, uc_handles_[i], mc_size_, 0);
    }

    detail::init_vas_for_handle(mc_handle_, device_ids_, num_devices, &uc_ptrs_, mc_size_);
    detail::init_vas_for_handle(mc_handle_, device_ids_, num_devices, &mc_ptrs_, mc_size_);

    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaMemset(reinterpret_cast<void*>(mc_ptrs_[i]), 0, mc_size_);
        cudaMemset(reinterpret_cast<void*>(uc_ptrs_[i]), 0, mc_size_);
    }
}

/**
 * @brief Destroy the hood object and clean up resources
 */
~hood() {
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(device_ids_[i]);
        cuMemUnmap(mc_ptrs_[i], mc_size_);
        cuMemUnmap(uc_ptrs_[i], mc_size_);
        cuMemAddressFree(mc_ptrs_[i], mc_size_);
        cuMemAddressFree(uc_ptrs_[i], mc_size_);
        cuMemRelease(uc_handles_[i]);
    }
    cuMemRelease(mc_handle_);
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
        if (cur_va_offset_ > mc_size_) {
            throw std::runtime_error("Exceeded multicast memory granularity");
        }
    } 
    return sync_id_to_va_offset_[sync_id];
}


public:
    const int num_devices;

private:
    int device_ids_[HOOD_SIZE];
    CUdeviceptr mc_ptrs_[HOOD_SIZE] = {};
    CUdeviceptr uc_ptrs_[HOOD_SIZE] = {};
    
    size_t mc_size_ = 0;
    size_t cur_va_offset_ = 0;
    
    CUmemGenericAllocationHandle mc_handle_ = {};
    CUmemGenericAllocationHandle uc_handles_[HOOD_SIZE] = {};
    
    std::unordered_map<int, size_t> sync_id_to_va_offset_;
};
} // namespace kittens 