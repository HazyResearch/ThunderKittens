#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory> 
#include "detail/helpers.cuh"
#include "../../common/common.cuh"

namespace kittens {

template <typename SYNC_SPACE_DTYPE = int>
struct sync_point {
    SYNC_SPACE_DTYPE *mc;
    SYNC_SPACE_DTYPE *uc;
};

namespace ducks {
namespace sync_manager {
    
struct identifier {};

/**
 * @brief Concept for all sync managers.
 */
template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace sync_manager
} // namespace ducks

/**
 * @brief The sync_manager struct manages multicast memory spaces across 
 * multiple GPUs for synchronization. It is just a named PGL under the hood.
 */
template <int NUM_DEVICES, int NUM_SYNC_POINTS = 16>
struct sync_manager {
    using identifier = ducks::sync_manager::identifier;
    
    static constexpr int sync_point_size = 3; // we need 3 points to sync (refer to gang::sync() for explanation)
    static constexpr int sync_space_size = NUM_SYNC_POINTS * sync_point_size;
    using SYNC_SPACE_DTYPE = int;
    using SYNC_SPACE_T = pgl<gl<SYNC_SPACE_DTYPE, 1, 1, 1, sync_space_size>, NUM_DEVICES, true>;
    SYNC_SPACE_T sync_space;

    // It is recommended to call the sync_manager.create() method instead of direct construction
    inline __host__ sync_manager(int *device_ids, SYNC_SPACE_DTYPE **d_sync_spaces) : 
        sync_space(device_ids, d_sync_spaces, nullptr, nullptr, nullptr, nullptr) { }

    static inline __host__ sync_manager create(int* device_ids) {
        SYNC_SPACE_DTYPE *d_sync_spaces[NUM_DEVICES]; 

        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_idx = device_ids[i];
            cudaSetDevice(dev_idx);
            pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_sync_spaces[dev_idx], sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }

        return sync_manager(device_ids, d_sync_spaces);
    }

    inline __host__ void free() {
        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_idx = sync_space.device_ids[i];
            cudaSetDevice(dev_idx);
            pglCudaFree(dev_idx, sync_space[i].raw_ptr, sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }
        pglFree(sync_space);
    }

    __device__ inline sync_point<SYNC_SPACE_DTYPE> get_sync_point(int sync_id, int dev_idx) const {
        return sync_point{
            sync_space.mc_vas[dev_idx] + sync_id * sync_point_size,
            sync_space[dev_idx].raw_ptr + sync_id * sync_point_size
        };
    }
};

} // namespace kittens