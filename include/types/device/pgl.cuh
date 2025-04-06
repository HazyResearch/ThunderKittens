/**
 * @file
 * @brief Templated layouts for parallel global memory.
 */

#pragma once

#include <iostream>
#include <algorithm>
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

/**
 * @brief Parallel global layouts. Represents a region of data spread across multiple devices.
 * @tparam GL The underlying global layout on each device.
 * @tparam NUM_DEVICES The number of GPU devices to use.
 * @tparam INIT_MC Whether to initialize the multicast handle inside the constructor.
 *             If false, the user must manually initialize the multicast handle. 
 * @tparam INIT_TMA Whether to initialize TMA descriptors for the PGL object (does not affect 
 *             TMA creation for underlying GLs). If false, the user can manually initialize later.
 * @tparam TMA_Types The types of TMA descriptors to use for the PGL object.
 */
template<kittens::ducks::gl::all GL, int NUM_DEVICES = 8, bool INIT_MC = true, bool INIT_TMA = false, typename... TMA_Types>
struct pgl {
    using identifier = ducks::pgl::identifier;
    using _GL = GL;
    using T = GL::dtype;
    using dtype = T;
    
    GL gls[NUM_DEVICES];
    
    size_t mc_size;     // size of the multicast handle
    CUmemGenericAllocationHandle mc_handle; // the single multicast handle for collective ops
    T *mc_vas[NUM_DEVICES];

    int device_ids[NUM_DEVICES];
    static constexpr int num_devices = NUM_DEVICES;

    detail::descriptor_dict<TMA_Types...> tma_descs[NUM_DEVICES];

    __host__ __device__ const GL &operator[](int idx) const { return gls[idx]; }
    __device__ inline T* mc_ptr_at(const coord<ducks::default_type> &idx, int dev_idx) const {
        const GL &gl = gls[dev_idx];
        return &mc_vas[dev_idx][((idx.b*gl.depth() + idx.d)*gl.rows() + idx.r)*gl.cols() + idx.c];
    }

    __host__ inline pgl(int *_device_ids,  // an array of NUM_DEVS device IDs
                        T **_data,         // an array of NUM_DEVS pointers
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
        pgl(std::make_index_sequence<NUM_DEVICES>{}, _device_ids, _data, _batch, _depth, _rows, _cols) { }

    template<size_t... I>
    __host__ inline pgl(std::index_sequence<I...>,
                        int *_device_ids,  // an array of NUM_DEVS device IDs
                        T **_data,         // an array of NUM_DEVS pointers
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
            gls{GL(_data[I], _batch, _depth, _rows, _cols)...}, mc_handle(0), mc_vas{} {
        static_assert(NUM_DEVICES > 1, 
            "SKILL ISSUE: No point in using pgl with a single device (CUDA doesn't allow it).");

        for (int i = 0; i < NUM_DEVICES; i++) {
            multicast_check(_device_ids[i]); // check if device supports multicast
            device_ids[i] = _device_ids[i];
        }

        if constexpr (INIT_MC) {
            multicast_init(); // should be called only once
            multicast_bind();
        }

        static_assert(INIT_MC || !INIT_TMA, 
            "Cannot auto-initialize TMA without auto-initializing multicast handle.");
        if constexpr (INIT_MC && INIT_TMA) tma_init();
    }

    // This should be only called once in the lifetime of the pgl object
    // There is a constraint on mc objects, which is apparently around 130 for 8 H100s on NVSwitch platform
    // NVIDIA does not document this and does not provide any driver APIs to clean resources during process runtime
    __host__ inline void multicast_init() {
        cuInit(0); // should be called before any Driver API calls (arg SBZ)

        if (mc_handle) {
            std::cerr << "Multicast handle already initialized." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Create MC handle props (once for all devices)
        CUmulticastObjectProp mc_prop = {}; 
        detail::init_mc_prop(&mc_prop, NUM_DEVICES, gl_size());
        mc_size = mc_prop.size;
        
        // Create MC handle (once for all devices)
        CUCHECK(cuMulticastCreate(&mc_handle, &mc_prop));
    
        // Add devices to MC handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUdevice dev;
            CUCHECK(cuDeviceGet(&dev, device_ids[i]));
            CUCHECK(cuMulticastAddDevice(mc_handle, dev)); // must be done before any bindig
        }

        // Get granularities
        size_t mc_granularity;
        cuMulticastGetGranularity(&mc_granularity, &mc_prop, MC_GRAN_TYPE);

        // Map virtual addresses to the mc handle. All devices must be added before any address can be mapped per CUDA docs
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(cuMemAddressReserve((CUdeviceptr *)&mc_vas[i], mc_size, mc_granularity, 0, 0));
            CUCHECK(cuMemMap((CUdeviceptr)mc_vas[i], mc_size, 0, mc_handle, 0));

            // Set access permissions
            CUmemAccessDesc desc = {};
            detail::init_mem_desc(&desc, device_ids[i]);
            CUCHECK(cuMemSetAccess((CUdeviceptr)mc_vas[i], mc_size, &desc, 1));
        }
    }

    // This can be called multiple times to share the pgl objects across different kernels. 
    // However, participating devices cannot be changed and the size must be less than or 
    // equal to the original size. GL dimensions can change (but must be the same across devices).
    // If you are calling this manually after constructor call, you should:
    //   1. Allocate device memory with pglCudaMalloc
    //   2. Create GL objects with the above device memory
    //   3. Set the GLs in this PGL object
    //   4. Call multicast_unbind() to unbind the old GLs
    //   5. Call multicast_bind() to bind the new GLs
    // If you call this multiple times without changing the GLs or unbinding the old bindings, runtime error will occur
    __host__ inline void multicast_bind() {
        if (!mc_handle) {
            std::cerr << "Multicast handle is uninitialized or destroyed." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Bind the underlying GLs (with memory alloc'ed by pglCudaMalloc) to the multicast handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(cuMulticastBindAddr(mc_handle, 0, (CUdeviceptr)gls[i].raw_ptr, mem_handle_size(gl_size()), 0));
        }
    }

    __host__ inline void tma_init() {
        if (!mc_handle) {
            std::cerr << "Multicast handle is uninitialized or destroyed." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            tma_descs[i] = detail::descriptor_dict<TMA_Types...>(
                mc_vas[i], gls[i].batch_internal, gls[i].depth_internal, gls[i].rows_internal, gls[i].cols_internal);
        }
    }

    __host__ inline void multicast_unbind() {
        if (!mc_handle) {
            std::cerr << "Multicast handle is uninitialized or destroyed." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Unbind the underlying GLs from the multicast handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev;
            cuDeviceGet(&dev, device_ids[i]);
            CUCHECK(cuMulticastUnbind(mc_handle, dev, 0, mem_handle_size(gl_size())));
        }
    }

    // Don't call this if you are reusing this pgl
    // Call multicast_unbind() first to unbind the old bindings
    // This only frees the v address space; CUDA does not provide a way to completely free the multicast handle
    __host__ inline void multicast_destroy() {
        if (!mc_handle) return; // already destroyed

        for (int i = 0; i < num_devices; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(cuMemUnmap((CUdeviceptr)mc_vas[i], mc_size));
            CUCHECK(cuMemAddressFree((CUdeviceptr)mc_vas[i], mc_size));
            mc_vas[i] = nullptr;
        }

        mc_handle = 0;
    }

    __host__ inline void multicast_check(int device_id) {
        cuInit(0); // should be called before any Driver API calls (arg SBZ)

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

    __host__ __device__ inline auto batch() const { return gls[0].batch(); }
    __host__ __device__ inline auto depth() const { return gls[0].depth(); }
    __host__ __device__ inline auto rows() const { return gls[0].rows(); }
    __host__ __device__ inline auto cols() const { return gls[0].cols(); }
    
    template<int axis> __device__ inline size_t shape() const { return gls[0].template shape<axis>(); }
    template<int axis> __device__ inline size_t stride() const { return gls[0].template stride<axis>(); }

    // We make this a function because users can change the underlying GL
    __host__ __device__ inline size_t gl_size() const {
        return static_cast<size_t>(this->batch()) * static_cast<size_t>(this->depth()) * 
               static_cast<size_t>(this->rows()) * static_cast<size_t>(this->cols()) * sizeof(T);
    }

    __host__ inline size_t mem_handle_size(size_t size) {
        size_t mem_granularity;
        CUmemAllocationProp mem_prop = {};
        CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, MEM_GRAN_TYPE));

        // This should be the size that was actually allocated to the handle
        if (size % mem_granularity != 0)
            size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;

        return size;
    }

    template<typename U, int axis> 
    __device__ inline const CUtensorMap* get_tma(int dev_idx) const {
        return tma_descs[dev_idx].template get<U, axis>();
    }
};

template <bool ROUND_UP = false, typename T>
__host__ inline void pglCudaMalloc(int num_devices, int* device_ids, int device_id, T **ptr, size_t size) {
    CUDACHECK(cudaSetDevice(device_id));

    // Create memory handle prop

    CUmemAllocationProp mem_prop = {}; 
    detail::init_mem_prop(&mem_prop, device_id);

    // Query for granularity
    size_t mem_granularity;
    CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, MEM_GRAN_TYPE));
    if (size % mem_granularity != 0) {
        if constexpr (ROUND_UP) {
            size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;
        } else {
            std::cerr << "ERROR (pglCudaMalloc): Size is not a multiple of the recommended granularity." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Allocate physical memory on the device
    CUmemGenericAllocationHandle mem_handle; // this can be retrieved with cuMemRetainAllocationHandle()
    CUCHECK(cuMemCreate(&mem_handle, size, &mem_prop, 0));

    // Bind virtual address
    CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, mem_granularity, 0, 0));
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, mem_handle, 0));

    // Set access
    CUmemAccessDesc desc_list[num_devices] = {};
    for (int i = 0; i < num_devices; i++)
        detail::init_mem_desc(&desc_list[i], device_ids[i]);
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, desc_list, num_devices));
}

template <typename T>
__host__ inline void pglCudaFree(int device_id, T *ptr, size_t size) {
    CUDACHECK(cudaSetDevice(device_id));

    // Query for granularity
    CUmemAllocationProp mem_prop = {}; 
    detail::init_mem_prop(&mem_prop, device_id);
    size_t mem_granularity;
    CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, MEM_GRAN_TYPE));

    // This should be the size that was actually allocated to the handle
    if (size % mem_granularity != 0)
        size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;

    // Retrieve handle
    CUmemGenericAllocationHandle mem_handle;
    CUCHECK(cuMemRetainAllocationHandle(&mem_handle, (void *)ptr)); 

    // Always free in this order
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size)); 
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CUCHECK(cuMemRelease(mem_handle));
}

// This should be called only once per pgl object
template<kittens::ducks::pgl::all PGL>
__host__ inline void pglFree(PGL &pgl_obj) {
    pgl_obj.multicast_unbind(); // unbind the old bindings
    pgl_obj.multicast_destroy(); // free the address space
}

} // namespace kittens
