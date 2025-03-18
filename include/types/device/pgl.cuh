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
}
}
    
template <kittens::ducks::gl::all GL>
struct PglObj {
    PglObj(GL _gl, typename GL::dtype *_mc_ptr, int _nelem, int _dev_id, int _num_devices)
        : gl(_gl), mc_ptr(_mc_ptr), nelem(_nelem), dev_id(_dev_id), num_devices(_num_devices) {}

    using T = GL::dtype;
    using dtype = T;
    GL gl;
    T *mc_ptr;
    // TODO: Need to figure out how to manage this
    int nelem;
    int dev_id;
    int num_devices;
};

// INIT_P: whether to initialize the multicast handle inside the constructor
//         if false, the user must manually initialize the multicast handle
//         in order to use TK collective operations
template<kittens::ducks::gl::all GL, bool INIT_P = true>
struct pgl {
    using identifier = ducks::pgl::identifier;
    
    using T = GL::dtype;
    using dtype = T;

    size_t size; // size of the raw data in bytes (on each device, not aggregate of all devices)
    std::vector<GL> tensors; // an array of global layouts
                              // should conceptually be a single tensor, shared across all devices

    CUmemGenericAllocationHandle mc_handle; // the single multicast handle for collective ops
    T **raw_multi_ptr;                 // mc_ptr, an array of virtual addresses on each machine attached to the mc_handle

    int nelem; // number of elements per device
    int num_devices; // number of CUDA devices
    int *device_ids; // CUDA device IDs. Corresponds to the order of vector<GL> tensors

    GL &operator[](int idx) { return tensors[idx]; } 

    __host__ inline pgl(int *_device_ids, // an array of NUM_DEVS device IDs
                        int _num_devices,  // number of devices
                        T **_data,        // an array of NUM_DEVS pointers
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols)
            : num_devices(_num_devices),
              device_ids(new int[_num_devices]),
              raw_multi_ptr(new T*[_num_devices]) {
        if (num_devices <= 1) {
            std::cerr << "SKILL ISSUE: No point in using pgl with a single device." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Ignore zeros (TODO: is this the best way to calculate size?)
        nelem = std::max<size_t>(size_t(_batch), 1) * std::max<size_t>(size_t(_depth), 1) *
                std::max<size_t>(size_t(_rows), 1) * std::max<size_t>(size_t(_cols), 1);
        size = nelem * sizeof(T);
    
        for (int i = 0; i < num_devices; i++) {
            multicast_check(_device_ids[i]); // check if device supports multicast
            device_ids[i] = _device_ids[i];
            tensors.emplace_back(_data[i], _batch, _depth, _rows, _cols); // construct each gl
        }

        if (INIT_P) {
            multicast_init();
        } else {
            mc_handle = 0;
            for (int i = 0; i < num_devices; i++) raw_multi_ptr = nullptr;
        }
    }

    // Users really shouldn't copy this object around
    // as it complicates CUDA virtual memory management
    pgl(const pgl &other) = delete;
    pgl& operator=(const pgl &other) = delete;

    __host__ inline ~pgl() {
        if (mc_handle) {
            for (int i = 0; i < num_devices; ++i) {
                cudaSetDevice(device_ids[i]);
                cuMemUnmap((CUdeviceptr)raw_multi_ptr[i], size);
                cuMemAddressFree((CUdeviceptr)raw_multi_ptr[i], size);
                cuMulticastUnbind(mc_handle, device_ids[i], 0, size);
            }
        }

        delete[] device_ids;
        delete[] raw_multi_ptr;
    }
    
    __host__ inline void multicast_init() {
        cuInit(0); // should be called before any Driver API calls (arg SBZ)
    
        // Create MC handle props (once for all devices)
        CUmulticastObjectProp mc_prop; 
        size_t mc_size = detail::init_mc_prop(&mc_prop, num_devices, size);
        
        // Create MC handle (once for all devices)
        cuMulticastCreate(&mc_handle, &mc_prop);
    
        // Add devices to MC handle
        for (int i = 0; i < num_devices; ++i) {
            CUdevice dev;
            cuDeviceGet(&dev, device_ids[i]);
            cuMulticastAddDevice(mc_handle, dev); // must be done before any bindig
        }

        // Attach MC handle to physical memory & map to virtual memory on each device
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(device_ids[i]);
            // Bind the physical memory (malloc'ed by user) to the multicast handle
            cuMulticastBindAddr(mc_handle, 0, (CUdeviceptr)tensors[i].raw_ptr, mc_size, 0);    
            cuMemAddressReserve((CUdeviceptr *)&raw_multi_ptr[i], mc_size, mc_size, 0, 0);
            cuMemMap((CUdeviceptr)raw_multi_ptr[i], mc_size, 0, mc_handle, 0);

            // Set access permissions
            CUmemAccessDesc desc = detail::create_mem_desc(device_ids[i]);
            cuMemSetAccess((CUdeviceptr)raw_multi_ptr[i], mc_size, &desc, 1);
        }
    }

    __host__ inline void multicast_check(int device_id) {
        // Check if device supports MultiCast Objects
        CUdevice dev;
        cuDeviceGet(&dev, device_id);
    
        int device_supports_multicast;
        cuDeviceGetAttribute(&device_supports_multicast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
    
        if (!device_supports_multicast) {
            std::cerr << "DEVICE ISSUE: Device " << device_id <<" does not support Multicast Objects." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    PglObj<GL> get_pgl_obj(int dev_id) {
        return {
            tensors[dev_id], 
            raw_multi_ptr[dev_id], // TODO: modify kernel example code to increment pointer on their end
            nelem, 
            dev_id,
            num_devices
        };
    }
};

namespace ducks {
namespace pgl {
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
}
}

template <typename T>
__host__ inline void pglCudaMalloc(int device_id, T **ptr, CUmemGenericAllocationHandle *mem_handle, size_t size) {
    cudaSetDevice(device_id);

    // Create memory handle prop
    CUmemAllocationProp mem_prop = detail::create_mem_prop(device_id);

    // Query for recommended granularity
    size_t mem_granularity;
    cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

    // Round up size to the nearest multiple of the granularity
    size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;

    // Allocate physical memory on the device
    cuMemCreate(mem_handle, size, &mem_prop, 0);

    // Bind virtual address
    cuMemAddressReserve((CUdeviceptr *)ptr, size, mem_granularity, 0, 0);
    cuMemMap((CUdeviceptr)*ptr, size, 0, *mem_handle, 0);

    // Set access
    CUmemAccessDesc desc = detail::create_mem_desc(device_id);
    cuMemSetAccess((CUdeviceptr)*ptr, size, &desc, 1);
}

template <typename T>
__host__ inline void pglCudaFree(int device_id, T *ptr, CUmemGenericAllocationHandle mem_handle, size_t size) {
    cudaSetDevice(device_id);
    cuMemUnmap((CUdeviceptr)ptr, size); // always free in this order
    cuMemAddressFree((CUdeviceptr)ptr, size);
    cuMemRelease(mem_handle);
}

} // namespace kittens
