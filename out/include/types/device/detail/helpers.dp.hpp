#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace kittens {

// We must ensure that we use the same granularity type for all functions
constexpr CUmemAllocationGranularity_flags_enum MEM_GRAN_TYPE =
    sycl::ext::oneapi::experimental::granularity_mode::recommended;
constexpr CUmulticastGranularity_flags_enum MC_GRAN_TYPE = CU_MULTICAST_GRANULARITY_RECOMMENDED;

namespace detail {
    // Returns size of the multicast granularity
    inline void init_mc_prop(CUmulticastObjectProp *mc_prop, int num_devices, size_t size) {
        mc_prop->numDevices = num_devices;
        mc_prop->handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
        mc_prop->flags = 0; // SBZ

        size_t granularity = 0;
        /*
        DPCT1007:460: Migration of cuMulticastGetGranularity is not supported.
        */
        cuMulticastGetGranularity(&granularity, mc_prop, kittens::MC_GRAN_TYPE);
        mc_prop->size = ((size + granularity - 1) / granularity) * granularity;
    }

    inline void init_mem_prop(dpct::experimental::mem_prop *mem_prop,
                              int device_id) {
        mem_prop->type = 0;
        mem_prop->location.type = 1;
        mem_prop->location.id = device_id;
        mem_prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    inline void init_mem_desc(dpct::experimental::mem_access_desc *desc,
                              int device_id) {
        desc->flags =
            sycl::ext::oneapi::experimental::address_access_mode::read_write;
        desc->location.id = device_id;
        desc->location.type = 1;
    }
} // namespace detail
} // namespace kittens