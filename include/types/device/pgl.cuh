/**
 * @file
 * @brief Templated layouts for parallel global memory.
 */

#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include "cuda.h"
#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "../global/global.cuh"
#include "detail/helpers.cuh"


namespace kittens {

/* ----------  Parallel global layout descriptor  ---------- */
// Refers to a single tensor shared across multiple devices
// Allows for the collective operations
namespace ducks {
namespace pgl {

struct identifier {};

/**
 * @brief Concept for all parallel global layouts.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as ducks::pgl::identifier.
 */
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace pgl
} // namespace ducks

template<kittens::ducks::gl::all GL, int NUM_DEVICES = 8, bool INIT_MC = true>
struct pgl {
    using identifier = ducks::pgl::identifier;
    using T = GL::dtype;
    using dtype = T;

    size_t size;        // size of the raw conceptual data in bytes (on each device, not aggregate of all devices)
    size_t handle_size; // size of actual memory 
    size_t mc_size;

    T *mc_vas[NUM_DEVICES];
    int device_ids[NUM_DEVICES];

    // A hack to avoid calling the gl default constructor
    alignas(GL) char _gls[NUM_DEVICES][sizeof(GL)];

    CUmemGenericAllocationHandle mc_handle; // the single multicast handle for collective ops
    size_t nelem;                           // number of elements per device

    __host__ __device__ GL &gls(int idx) { return *reinterpret_cast<GL*>(&_gls[idx]); } 
    __host__ __device__ GL &operator[](int idx) { return gls(idx); } 

    __host__ __device__ const GL &gls(int idx) const { return *reinterpret_cast<const GL*>(&_gls[idx]); }
    __host__ __device__ const GL &operator[](int idx) const { return gls(idx); }

    __host__ inline pgl(int *_device_ids,  // an array of NUM_DEVS device IDs
                        T **_data,         // an array of NUM_DEVS pointers
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) {
        if constexpr (NUM_DEVICES <= 1) {
            std::cerr << "SKILL ISSUE: No point in using pgl with a single device." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        nelem = std::max<size_t>(size_t(_batch), 1) * std::max<size_t>(size_t(_depth), 1) *
                std::max<size_t>(size_t(_rows), 1) * std::max<size_t>(size_t(_cols), 1);
        size = nelem * sizeof(T);

        CUmemAllocationProp mem_prop{};
        size_t mem_granularity;
        CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        handle_size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;
    
        for (int i = 0; i < NUM_DEVICES; i++) {
            multicast_check(_device_ids[i]); // check if device supports multicast
            device_ids[i] = _device_ids[i];
            new (&_gls[i]) GL(_data[i], _batch, _depth, _rows, _cols);
        }

        if (INIT_MC) {
            multicast_init();
        } else {
            mc_handle = 0;
            for (int i = 0; i < NUM_DEVICES; i++) mc_vas[i] = nullptr;
        }
    }

    // Device code should be able to call this, but not actually do anything
    // #ifdef __CUDA_ARCH__
    // __device__ ~pgl() { }
    // #else
    // __host__ ~pgl() {
    //     // Host-only logic
    //     for (int i = 0; i < NUM_DEVICES; i++) {
    //         CUDACHECK(cudaSetDevice(device_ids[i]));
    //         if (mc_handle) {
    //             CUCHECK(cuMemUnmap((CUdeviceptr)mc_vas[i], mc_size));
    //             CUCHECK(cuMemAddressFree((CUdeviceptr)mc_vas[i], mc_size));
    //             CUCHECK(cuMulticastUnbind(mc_handle, device_ids[i], 0, handle_size));
    //         }
    //     }
    // }
    // #endif

    __host__ inline void multicast_init() {
        cuInit(0); // should be called before any Driver API calls (arg SBZ)
    
        // Create MC handle props (once for all devices)
        CUmulticastObjectProp mc_prop; 
        mc_size = detail::init_mc_prop(&mc_prop, NUM_DEVICES, size);
        
        // Create MC handle (once for all devices)
        CUCHECK(cuMulticastCreate(&mc_handle, &mc_prop));
    
        // Add devices to MC handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUdevice dev;
            CUCHECK(cuDeviceGet(&dev, device_ids[i]));
            CUCHECK(cuMulticastAddDevice(mc_handle, dev)); // must be done before any bindig
        }

        size_t mc_granularity = 0;
        cuMulticastGetGranularity(&mc_granularity, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);

        // Attach MC handle to physical memory & map to virtual memory on each device
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            // Bind the physical memory (malloc'ed by user) to the multicast handle
            CUCHECK(cuMulticastBindAddr(mc_handle, 0, (CUdeviceptr)gls(i).raw_ptr, handle_size, 0));

            CUCHECK(cuMemAddressReserve((CUdeviceptr *)&mc_vas[i], mc_size, mc_granularity, 0, 0));
            CUCHECK(cuMemMap((CUdeviceptr)mc_vas[i], mc_size, 0, mc_handle, 0));

            // Set access permissions
            CUmemAccessDesc desc = detail::create_mem_desc(device_ids[i]);
            CUCHECK(cuMemSetAccess((CUdeviceptr)mc_vas[i], mc_size, &desc, 1));
        }
    }

    __host__ inline void multicast_check(int device_id) {
        // Check if device supports MultiCast Objects
        CUdevice dev;
        CUCHECK(cuDeviceGet(&dev, device_id));
    
        int device_supports_multicast;
        CUCHECK(cuDeviceGetAttribute(&device_supports_multicast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    
        if (!device_supports_multicast) {
            std::cerr << "DEVICE ISSUE: Device " << device_id <<" does not support Multicast Objects." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
};

template <bool ROUND_UP = false, typename T>
__host__ inline void pglCudaMalloc(int num_devices, int* device_ids, int device_id, T **ptr, CUmemGenericAllocationHandle *mem_handle, size_t size) {
    CUDACHECK(cudaSetDevice(device_id));

    // Create memory handle prop
    CUmemAllocationProp mem_prop = detail::create_mem_prop(device_id);

    // Query for recommended granularity
    size_t mem_granularity;
    CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    if (size % mem_granularity != 0) {
        if constexpr (ROUND_UP) {
            size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;
        } else {
            std::cerr << "ERROR (pglCudaMalloc): Size is not a multiple of the recommended granularity." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Allocate physical memory on the device
    CUCHECK(cuMemCreate(mem_handle, size, &mem_prop, 0));

    // Bind virtual address
    CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, mem_granularity, 0, 0));
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, *mem_handle, 0));

    // Set access
    CUmemAccessDesc desc_list[num_devices];
    for (int i = 0; i < num_devices; i++) {
        desc_list[i] = detail::create_mem_desc(device_ids[i]);
    }
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, desc_list, num_devices));
}

template <typename T>
__host__ inline void pglCudaFree(int device_id, T *ptr, CUmemGenericAllocationHandle mem_handle, size_t size) {
    CUDACHECK(cudaSetDevice(device_id));
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size)); // always free in this order
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CUCHECK(cuMemRelease(mem_handle));
}

} // namespace kittens
