#pragma once 

#include <cuda.h>

namespace kittens {

// We must ensure that we use the same granularity type for all functions
constexpr CUmemAllocationGranularity_flags_enum MEM_GRAN_TYPE = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
constexpr CUmulticastGranularity_flags_enum MC_GRAN_TYPE = CU_MULTICAST_GRANULARITY_RECOMMENDED;

namespace detail {
    // Returns size of the multicast granularity
    __host__ inline void init_mc_prop(CUmulticastObjectProp *mc_prop, int num_devices, size_t size) {
        mc_prop->numDevices = num_devices;
        mc_prop->handleTypes = CU_MEM_HANDLE_TYPE_NONE; // single node
        mc_prop->flags = 0; // SBZ

        size_t granularity = 0;
        cuMulticastGetGranularity(&granularity, mc_prop, kittens::MC_GRAN_TYPE);
        mc_prop->size = ((size + granularity - 1) / granularity) * granularity;
    }

    __host__ inline size_t get_mc_granularity(int num_devices) {
        CUmulticastObjectProp mc_prop = {};
        mc_prop.numDevices = num_devices;
        mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_NONE; // single node
        mc_prop.flags = 0; // SBZ

        size_t granularity = 0;
        cuMulticastGetGranularity(&granularity, &mc_prop, kittens::MC_GRAN_TYPE);
        return granularity;
    }

    __host__ inline void init_mem_prop(CUmemAllocationProp *mem_prop, int device_id) {
        mem_prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
        mem_prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        mem_prop->location.id = device_id;
        mem_prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    }

    __host__ inline void init_mem_desc(CUmemAccessDesc *desc, int device_id) {
        desc->flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc->location.id = device_id;
        desc->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    }

    __host__ inline size_t get_mem_granularity(int device_id) {
        CUmemAllocationProp mem_prop = {}; 
        init_mem_prop(&mem_prop, device_id);
        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &mem_prop, MEM_GRAN_TYPE);
        return granularity;
    }
} // namespace detail
} // namespace kittens
