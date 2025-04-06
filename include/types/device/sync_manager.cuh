#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory> 
#include "detail/helpers.cuh"
#include "../../common/common.cuh"

namespace kittens {
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

template <typename SYNC_SPACE_DTYPE = int>
struct block_sync_point {
    SYNC_SPACE_DTYPE *mc;
    SYNC_SPACE_DTYPE *uc;
};

template <typename SYNC_SPACE_DTYPE = int>
struct grid_sync_point {
    SYNC_SPACE_DTYPE *gang_mc;
    SYNC_SPACE_DTYPE *gang_uc;
    SYNC_SPACE_DTYPE *grid_uc;
};

/**
 * @brief The sync_manager struct manages multicast memory spaces across 
 * multiple GPUs for synchronization. It is just a named PGL under the hood.
 * 
 * If you want to call gang::grid_sync(...), MAX_BLOCKS must be -1.
 * If you want to call gang::block_sync(...), set MAX_BLOCKS to the 
 * maximum number of blocks you expect to use on any device.
 */
template <int NUM_DEVICES, int MAX_BLOCKS = -1, int MAX_SYNC_POINTS = 4>
struct sync_manager {
    using identifier = ducks::sync_manager::identifier;

    static_assert(NUM_DEVICES > 0, "NUM_DEVICES must be greater than 0");
    static_assert(MAX_BLOCKS == -1 || MAX_BLOCKS > 0, "MAX_BLOCKS must be greater than 0 or -1");
    static_assert(MAX_SYNC_POINTS > 0, "MAX_SYNC_POINTS must be greater than 0");

    static constexpr bool is_grid_sync = MAX_BLOCKS == -1;
    static constexpr int sync_point_size = is_grid_sync ? 2 : MAX_BLOCKS;
    static constexpr int max_sync_points = MAX_SYNC_POINTS;
    static constexpr int sync_space_size = sync_point_size * max_sync_points;

    using SYNC_SPACE_DTYPE = int;
    using SYNC_SPACE_T = pgl<gl<SYNC_SPACE_DTYPE, 1, 1, 1, sync_space_size>, NUM_DEVICES, true>;
    SYNC_SPACE_T sync_space;

    // We recommend calling sync_manager.create() instead of direct construction
    inline __host__ sync_manager(int *device_ids, SYNC_SPACE_DTYPE **d_sync_spaces) : 
        sync_space(device_ids, d_sync_spaces, nullptr, nullptr, nullptr, nullptr) { }

    static inline __host__ sync_manager create(int* device_ids) {
        SYNC_SPACE_DTYPE *d_sync_spaces[NUM_DEVICES]; 

        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_id = device_ids[i];
            cudaSetDevice(dev_id);
            pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_id, &d_sync_spaces[i], sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }

        return sync_manager(device_ids, d_sync_spaces);
    }

    inline __host__ void free() {
        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_id = sync_space.device_ids[i];
            cudaSetDevice(dev_id);
            pglCudaFree(dev_id, sync_space[i].raw_ptr, sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }
        pglFree(sync_space);
    }

    /*
     * Get the sync point for a specific sync ID, device index, and block index.
     * This is used for block-level synchronization. (gang::block_sync)
     */
    __device__ inline block_sync_point<SYNC_SPACE_DTYPE> get_sync_point(const int sync_id, const int dev_idx, const int block_idx) const {
        static_assert(!is_grid_sync, "You cannot specify block_idx for grid-level sync");
        return block_sync_point{
            sync_space.mc_vas[dev_idx] + sync_id * sync_point_size + block_idx,
            sync_space[dev_idx].raw_ptr + sync_id * sync_point_size + block_idx,
        };
    }

    /*
     * Get the sync point for a specific sync ID and device index.
     * This is used for grid-level synchronization. (gang::grid_sync)
     */
    __device__ inline grid_sync_point<SYNC_SPACE_DTYPE> get_sync_point(const int sync_id, const int dev_idx) const {
        static_assert(is_grid_sync, "You need to specify block_idx for block-level sync");
        return grid_sync_point{
            sync_space.mc_vas[dev_idx] + sync_id * sync_point_size,
            sync_space[dev_idx].raw_ptr + sync_id * sync_point_size,
            sync_space[dev_idx].raw_ptr + sync_id * sync_point_size + 1,
        };
    }
};

} // namespace kittens