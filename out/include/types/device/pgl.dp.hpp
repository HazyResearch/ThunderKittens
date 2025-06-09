/**
 * @file
 * @brief Templated layouts for parallel global memory.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <algorithm>
#include "../../common/common.dp.hpp"
#include "../shared/shared.dp.hpp"
#include "../global/global.dp.hpp"
#include "detail/helpers.dp.hpp"

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
    dpct::experimental::physical_mem_ptr
        mc_handle; // the single multicast handle for collective ops
    T *mc_vas[NUM_DEVICES];

    int device_ids[NUM_DEVICES];
    static constexpr int num_devices = NUM_DEVICES;

    detail::descriptor_dict<TMA_Types...> tma_descs[NUM_DEVICES];

    const GL &operator[](int idx) const { return gls[idx]; }
    inline T* mc_ptr_at(const coord<ducks::default_type> &idx, int dev_idx) const {
        const GL &gl = gls[dev_idx];
        return &mc_vas[dev_idx][((idx.b*gl.depth() + idx.d)*gl.rows() + idx.r)*gl.cols() + idx.c];
    }

    inline pgl(int *_device_ids,  // an array of NUM_DEVS device IDs
                        T **_data,         // an array of NUM_DEVS pointers
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
        pgl(std::make_index_sequence<NUM_DEVICES>{}, _device_ids, _data, _batch, _depth, _rows, _cols) { }

    template<size_t... I>
    inline pgl(std::index_sequence<I...>,
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
    inline void multicast_init() {
        /*
        DPCT1026:461: The call to cuInit was removed because this functionality
        is redundant in SYCL.
        */
; // should be called before any Driver API calls (arg SBZ)

        if (mc_handle) {
            std::cerr << "Multicast handle already initialized." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Create MC handle props (once for all devices)
        CUmulticastObjectProp mc_prop = {}; 
        detail::init_mc_prop(&mc_prop, NUM_DEVICES, gl_size());
        mc_size = mc_prop.size;
        
        // Create MC handle (once for all devices)
        /*
        DPCT1007:465: Migration of cuMulticastCreate is not supported.
        */
        CUCHECK(cuMulticastCreate(&mc_handle, &mc_prop));

        // Add devices to MC handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev;
            CUCHECK(cuDeviceGet(&dev, device_ids[i]));
            /*
            DPCT1007:467: Migration of cuMulticastAddDevice is not supported.
            */
            CUCHECK(cuMulticastAddDevice(
                mc_handle, dev)); // must be done before any bindig
        }

        // Get granularities
        size_t mc_granularity;
        /*
        DPCT1007:462: Migration of cuMulticastGetGranularity is not supported.
        */
        cuMulticastGetGranularity(&mc_granularity, &mc_prop, MC_GRAN_TYPE);

        // Map virtual addresses to the mc handle. All devices must be added before any address can be mapped per CUDA docs
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(DPCT_CHECK_ERROR(
                *(dpct::device_ptr *)&mc_vas[i] = (dpct::device_ptr)
                    sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        (uintptr_t)0, mc_size,
                        dpct::get_current_device().get_context())));
            CUCHECK(DPCT_CHECK_ERROR((mc_handle)->map(
                (uintptr_t)(dpct::device_ptr)mc_vas[i], mc_size,
                sycl::ext::oneapi::experimental::address_access_mode::
                    read_write,
                0)));

            // Set access permissions
            dpct::experimental::mem_access_desc desc = {};
            detail::init_mem_desc(&desc, device_ids[i]);
            CUCHECK(DPCT_CHECK_ERROR(
                sycl::ext::oneapi::experimental::set_access_mode(
                    (dpct::device_ptr)mc_vas[i], mc_size, desc.flags,
                    dpct::get_device(desc.location.id).get_context())));
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
    inline void multicast_bind() {
        if (!mc_handle) {
            std::cerr << "Multicast handle is uninitialized or destroyed." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Bind the underlying GLs (with memory alloc'ed by pglCudaMalloc) to the multicast handle
        for (int i = 0; i < NUM_DEVICES; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(cuMulticastBindAddr(mc_handle, 0,
                                        (dpct::device_ptr)gls[i].raw_ptr,
                                        mem_handle_size(gl_size()), 0));
        }
    }

    inline void tma_init() {
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

    inline void multicast_unbind() {
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
    inline void multicast_destroy() {
        if (!mc_handle) return; // already destroyed

        for (int i = 0; i < num_devices; ++i) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            CUCHECK(DPCT_CHECK_ERROR(sycl::ext::oneapi::experimental::unmap(
                (dpct::device_ptr)mc_vas[i], mc_size,
                dpct::get_current_device().get_context())));
            CUCHECK(DPCT_CHECK_ERROR(
                sycl::ext::oneapi::experimental::free_virtual_mem(
                    (uintptr_t)(dpct::device_ptr)mc_vas[i], mc_size,
                    dpct::get_current_device().get_context())));
            mc_vas[i] = nullptr;
        }

        mc_handle = 0;
    }

    inline void multicast_check(int device_id) {
        /*
        DPCT1026:469: The call to cuInit was removed because this functionality
        is redundant in SYCL.
        */
; // should be called before any Driver API calls (arg SBZ)

        // Check if device supports MultiCast Objects
        int dev;
        CUCHECK(DPCT_CHECK_ERROR(dev = device_id));

        int device_supports_multicast;
        /*
        DPCT1028:470: The cuDeviceGetAttribute was not migrated because
        parameter CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED is unsupported.
        */
        CUCHECK(cuDeviceGetAttribute(&device_supports_multicast,
                                     CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                                     dev));

        if (!device_supports_multicast) {
            std::cerr << "DEVICE ISSUE: Device " << device_id <<" does not support Multicast Objects." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    inline auto batch() const { return gls[0].batch(); }
    inline auto depth() const { return gls[0].depth(); }
    inline auto rows() const { return gls[0].rows(); }
    inline auto cols() const { return gls[0].cols(); }
    
    template<int axis> inline size_t shape() const { return gls[0].template shape<axis>(); }
    template<int axis> inline size_t stride() const { return gls[0].template stride<axis>(); }

    // We make this a function because users can change the underlying GL
    inline size_t gl_size() const {
        return static_cast<size_t>(this->batch()) * static_cast<size_t>(this->depth()) * 
               static_cast<size_t>(this->rows()) * static_cast<size_t>(this->cols()) * sizeof(T);
    }

    inline size_t mem_handle_size(size_t size) {
        size_t mem_granularity;
        dpct::experimental::mem_prop mem_prop = {};
        CUCHECK(DPCT_CHECK_ERROR(
            mem_granularity =
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    dpct::get_device(mem_prop.location.id),
                    dpct::get_device(mem_prop.location.id).get_context(),
                    MEM_GRAN_TYPE)));

        // This should be the size that was actually allocated to the handle
        if (size % mem_granularity != 0)
            size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;

        return size;
    }

    template<typename U, int axis> 
    inline const CUtensorMap* get_tma(int dev_idx) const {
        return tma_descs[dev_idx].template get<U, axis>();
    }
};

template <bool ROUND_UP = false, typename T>
inline void pglCudaMalloc(int num_devices, int *device_ids, int device_id,
                          T **ptr, size_t size) try {
    /*
    DPCT1093:471: The "device_id" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    CUDACHECK(DPCT_CHECK_ERROR(dpct::select_device(device_id)));

    // Create memory handle prop

    dpct::experimental::mem_prop mem_prop = {};
    detail::init_mem_prop(&mem_prop, device_id);

    // Query for granularity
    size_t mem_granularity;
    CUCHECK(DPCT_CHECK_ERROR(
        mem_granularity = sycl::ext::oneapi::experimental::get_mem_granularity(
            dpct::get_device(mem_prop.location.id),
            dpct::get_device(mem_prop.location.id).get_context(),
            MEM_GRAN_TYPE)));
    if (size % mem_granularity != 0) {
        if constexpr (ROUND_UP) {
            size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;
        } else {
            std::cerr << "ERROR (pglCudaMalloc): Size is not a multiple of the recommended granularity." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Allocate physical memory on the device
    dpct::experimental::physical_mem_ptr
        mem_handle; // this can be retrieved with cuMemRetainAllocationHandle()
    CUCHECK(DPCT_CHECK_ERROR(
        mem_handle = new sycl::ext::oneapi::experimental::physical_mem(
            dpct::get_device(mem_prop.location.id),
            dpct::get_device(mem_prop.location.id).get_context(), size)));

    // Bind virtual address
    CUCHECK(DPCT_CHECK_ERROR(
        *(dpct::device_ptr *)ptr = (dpct::device_ptr)
            sycl::ext::oneapi::experimental::reserve_virtual_mem(
                (uintptr_t)0, size, dpct::get_current_device().get_context())));
    CUCHECK(DPCT_CHECK_ERROR(mem_handle->map(
        (uintptr_t)(dpct::device_ptr)*ptr, size,
        sycl::ext::oneapi::experimental::address_access_mode::read_write, 0)));

    // Set access
    dpct::experimental::mem_access_desc desc_list[num_devices] = {};
    for (int i = 0; i < num_devices; i++)
        detail::init_mem_desc(&desc_list[i], device_ids[i]);
    CUCHECK(DPCT_CHECK_ERROR(sycl::ext::oneapi::experimental::set_access_mode(
        (dpct::device_ptr)*ptr, size, desc_list->flags,
        dpct::get_device(desc_list->location.id).get_context())));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
inline void pglCudaFree(int device_id, T *ptr, size_t size) try {
    /*
    DPCT1093:472: The "device_id" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    CUDACHECK(DPCT_CHECK_ERROR(dpct::select_device(device_id)));

    // Query for granularity
    dpct::experimental::mem_prop mem_prop = {};
    detail::init_mem_prop(&mem_prop, device_id);
    size_t mem_granularity;
    CUCHECK(DPCT_CHECK_ERROR(
        mem_granularity = sycl::ext::oneapi::experimental::get_mem_granularity(
            dpct::get_device(mem_prop.location.id),
            dpct::get_device(mem_prop.location.id).get_context(),
            MEM_GRAN_TYPE)));

    // This should be the size that was actually allocated to the handle
    if (size % mem_granularity != 0)
        size = ((size + mem_granularity - 1) / mem_granularity) * mem_granularity;

    // Retrieve handle
    dpct::experimental::physical_mem_ptr mem_handle;
    /*
    DPCT1007:473: Migration of cuMemRetainAllocationHandle is not supported.
    */
    CUCHECK(cuMemRetainAllocationHandle(&mem_handle, (void *)ptr));

    // Always free in this order
    CUCHECK(DPCT_CHECK_ERROR(sycl::ext::oneapi::experimental::unmap(
        (dpct::device_ptr)ptr, size,
        dpct::get_current_device().get_context())));
    CUCHECK(DPCT_CHECK_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
        (uintptr_t)(dpct::device_ptr)ptr, size,
        dpct::get_current_device().get_context())));
    CUCHECK(DPCT_CHECK_ERROR(delete (mem_handle)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// This should be called only once per pgl object
template<kittens::ducks::pgl::all PGL>
inline void pglFree(PGL &pgl_obj) {
    pgl_obj.multicast_unbind(); // unbind the old bindings
    pgl_obj.multicast_destroy(); // free the address space
}

} // namespace kittens
