#pragma once 

#include <cuda.h>

namespace kittens {
namespace detail {
    // Returns size of the multicast granularity
    size_t init_mc_prop(CUmulticastObjectProp *mc_prop, int num_devices, size_t size = 0) {
        printf("Initial size: %zu\n", size);
        mc_prop->numDevices = num_devices;
        mc_prop->handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
        mc_prop->flags = 0; // SBZ
        
        // If size is not provided, return the recommended granularity
        size_t mc_size = 0;
        cuMulticastGetGranularity(&mc_size, mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
        // cuMulticastGetGranularity(&mc_size, mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM);
        if (size != 0) mc_size = ((size + mc_size - 1) / mc_size) * mc_size;
        
        mc_prop->size = mc_size;
        printf("Final size: %zu\n", mc_size);
        return mc_size;
    }

    CUmemAllocationProp create_mem_prop(int device_id) {
        CUmemAllocationProp memProp = {};
        memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        memProp.location.id = device_id;
        memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        return memProp;
    }

    CUmemAccessDesc create_mem_desc(int device_id) {
        CUmemAccessDesc desc = {}; 
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc.location.id = device_id;
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        return desc;
    }
} // namespace detail
} // namespace kittens