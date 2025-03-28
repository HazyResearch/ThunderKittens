#pragma once 

#include <cuda.h>

namespace kittens {

// We must ensure that we use the same granularity type for all functions
constexpr CUmemAllocationGranularity_flags_enum MEM_GRAN_TYPE = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
constexpr CUmulticastGranularity_flags_enum MC_GRAN_TYPE = CU_MULTICAST_GRANULARITY_RECOMMENDED;

namespace detail {
    // Returns size of the multicast granularity
    __host__ inline size_t init_mc_prop(CUmulticastObjectProp *mc_prop, int num_devices, size_t size = 0) {
        mc_prop->numDevices = num_devices;
        mc_prop->handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
        mc_prop->flags = 0; // SBZ
        
        // If size is not provided, return the recommended granularity (TODO: separate this functionality)
        size_t mc_size = 0;
        cuMulticastGetGranularity(&mc_size, mc_prop, kittens::MC_GRAN_TYPE);
        // cuMulticastGetGranularity(&mc_size, mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM);
        if (size != 0) mc_size = ((size + mc_size - 1) / mc_size) * mc_size;
        
        mc_prop->size = mc_size;
        return mc_size;
    }

    __host__ inline CUmemAllocationProp create_mem_prop(int device_id) {
        CUmemAllocationProp memProp = {};
        memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        memProp.location.id = device_id;
        memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        return memProp;
    }

    __host__ inline CUmemAccessDesc create_mem_desc(int device_id) {
        CUmemAccessDesc desc = {}; 
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc.location.id = device_id;
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        return desc;
    }
} // namespace detail
} // namespace kittens