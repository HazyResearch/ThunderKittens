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
#include "gl.cuh"
#include "util.cuh"

namespace kittens {

/* ----------  Parallel global layout descriptor  ---------- */
// Refers to a single tensor shared across multiple devices
// Allows for the collective operations

// INIT_P: whether to initialize the multicast handle inside the constructor
//         if false, the user must manually initialize the multicast handle
//         in order to use TK collective operations
template<kittens::ducks::gl::all GL, bool INIT_P = true>
struct pgl {
    using T = GL::dtype;

    size_t size; // size of the raw data in bytes (on each device, not aggregate of all devices)
    std::vector<GL> tensors; // an array of global layouts
                              // should conceptually be a single tensor, shared across all devices

    CUmemGenericAllocationHandle multi_handle; // the single multicast handle for collective ops
    T **raw_multi_ptr;                 // an array of virtual addresses on each machine attached to the multi_handle

    int num_devices; // number of CUDA devices
    int *device_ids; // CUDA device IDs. Corresponds to the order of vector<GL> tensors

    // Convience operator to easily use normal GL-targetted operations
    GL &operator[](int idx) { return tensors[idx]; } 

    /*

        Expected construction flow:

            using  base_tile        =  st_bf<64, 64>;
            using  global_layout    =  gl<bf16, 1, 1, -1, -1, base_tile>;
            using  pglobal_layout   =  pgl<gl<bf16, 1, 1, -1, -1, base_tile>, true>;

            <A, B allocations>

            __nv_bfloat16 *d_Cs[NUM_DEVS];
            for (int i = 0; i < NUM_DEVS; i++) {
                cudaSetDevice(i);
                cudaMalloc(&d_Cs[i], M * N * sizeof(__nv_bfloat16));
            }

            struct globals { 
                global_layout  A, B; 
                pglobal_layout C; 
            };

            int device_ids[NUM_DEVS] = {0, 1, 2, 3};
            pglobal_layout Cpg{device_ids, d_Cs, nullptr, nullptr, M, N};

            globals G{Ag, Bg, Cpg};
            some_kernel<<< ... >>>(G);

            

            normal_op(Cpg[i], ...);
            reduce_op(Cpg, ...);
    */

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
        size = std::max<size_t>(size_t(_batch), 1) * std::max<size_t>(size_t(_depth), 1) *
               std::max<size_t>(size_t(_rows), 1) * std::max<size_t>(size_t(_cols), 1) * sizeof(T);

        for (int i = 0; i < num_devices; i++) {
            multicast_check(_device_ids[i]); // check if device supports multicast
            device_ids[i] = _device_ids[i];
            tensors.emplace_back(_data[i], _batch, _depth, _rows, _cols); // construct each gl
        }

        if (INIT_P) {
            multicast_init();
        } else {
            multi_handle = 0;
            for (int i = 0; i < num_devices; i++) raw_multi_ptr = nullptr;
        }
    }

    // Copy Constructor
    // Users really shouldn't copy this object around
    // as it complicates CUDA virtual memory management
    __host__ inline pgl(const pgl &other):
        size(other.size),
        num_devices(other.num_devices),
        tensors(other.tensors),
        device_ids(new int[other.num_devices]),
        raw_multi_ptr(new T*[other.num_devices]),
        multi_handle(other.multi_handle)
    {
        for (int i = 0; i < num_devices; i++) {
            device_ids[i] = other.device_ids[i];
            raw_multi_ptr[i] = other.raw_multi_ptr[i];
        }
    }

    __host__ inline ~pgl() {
        delete[] device_ids;
        delete[] raw_multi_ptr;

        if (multi_handle) {
            for (int i = 0; i < num_devices; ++i) {
                int device_id = device_ids[i];
                cudaSetDevice(device_id);

                // Always free in this order
                cuMemUnmap((CUdeviceptr)raw_multi_ptr[i], size);
                cuMemAddressFree((CUdeviceptr)raw_multi_ptr[i], size);

                int dev;
                cuDeviceGet(&dev, device_id);
                cuMulticastUnbind(multi_handle, dev, 0, size);
            }
        }
    }
    
    __host__ inline void multicast_init() {
        cuInit(0); // should be called before any Driver API calls (arg SBZ)
    
        // Create MC handle props (once for all devices)
        CUmulticastObjectProp mc_prop = {};
        mc_prop.numDevices = num_devices;
        mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
        mc_prop.flags = 0; // SBZ
    
        // Query for granularities
        size_t granularity = 0;
        cuMulticastGetGranularity(&granularity, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
        if (size % granularity != 0) {
            std::cerr << "SKILL ISSUE: pgl size must be a multiple of granularity " << granularity << std::endl;
            std::exit(EXIT_FAILURE);
        }
        mc_prop.size = size;
    
        // Create MC handle (once for all devices)
        cuMulticastCreate(&multi_handle, &mc_prop);
    
        // Add devices to MC handle
        for (int i = 0; i < num_devices; ++i) {
            CUdevice dev;
            cuDeviceGet(&dev, device_ids[i]);
            cuMulticastAddDevice(multi_handle, dev); // must be done before any bindig
        }

        // Attach MC handle to physical memory & map to virtual memory on each device
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(device_ids[i]);

            // Bind the physical memory (malloc'ed by user) to the multicast handle
            cuMulticastBindAddr(multi_handle, 0, (CUdeviceptr)tensors[i].raw_ptr, size, 0);
            
            // Create virtual address for the multicast handle
            cuMemAddressReserve((CUdeviceptr *)&raw_multi_ptr[i], size, granularity, 0, 0);
            cuMemMap((CUdeviceptr)raw_multi_ptr[i], size, 0, multi_handle, 0);

            // Set access permissions
            CUmemAccessDesc desc[1];
            desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            desc[0].location.id = device_ids[i];
            desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            cuMemSetAccess((CUdeviceptr)raw_multi_ptr[i], size, desc, 1);
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
};

template <typename T>
__host__ inline void pglCudaMalloc(int device_id, T **ptr, CUmemGenericAllocationHandle *mem_handle, size_t size) {
    cudaSetDevice(device_id);

    // Create memory handle prop
    CUmemAllocationProp mem_prop = {};
    mem_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // use pinned memory
    mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mem_prop.location.id = device_id;
    mem_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Query for recommended granularity
    size_t mem_granularity;
    cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (size % mem_granularity != 0) {
        std::cerr << "pglCudaMalloc: Size must be a multiple of mem granularity " << mem_granularity << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Allocate physical memory on the device
    cuMemCreate(mem_handle, size, &mem_prop, 0);

    // Bind virtual address
    cuMemAddressReserve((CUdeviceptr *)ptr, size, mem_granularity, 0, 0);
    cuMemMap((CUdeviceptr)*ptr, size, 0, *mem_handle, 0);

    // Set access
    CUmemAccessDesc desc[1];
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc[0].location.id = device_id;
    desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    cuMemSetAccess((CUdeviceptr)*ptr, size, desc, 1);
}

template <typename T>
__host__ inline void pglCudaFree(int device_id, T *ptr, CUmemGenericAllocationHandle mem_handle, size_t size) {
    cudaSetDevice(device_id);
    cuMemUnmap((CUdeviceptr)ptr, size); // always free in this order
    cuMemAddressFree((CUdeviceptr)ptr, size);
    cuMemRelease(mem_handle);
}

}
