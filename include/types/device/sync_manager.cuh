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

enum sync_level {
    GRID = 0,
    BLOCK = 1
};

template <typename SYNC_SPACE_DTYPE, sync_level LEVEL>
struct sync_point {
    static_assert(std::is_same_v<SYNC_SPACE_DTYPE, std::true_type>, "Invalid sync level.");
};
template <typename SYNC_SPACE_DTYPE>
struct sync_point<SYNC_SPACE_DTYPE, sync_level::BLOCK> {
    SYNC_SPACE_DTYPE *gang_mc;
    SYNC_SPACE_DTYPE *gang_uc;
};
template <typename SYNC_SPACE_DTYPE>
struct sync_point<SYNC_SPACE_DTYPE, sync_level::GRID> {
    SYNC_SPACE_DTYPE *gang_mc;
    SYNC_SPACE_DTYPE *gang_uc;
    SYNC_SPACE_DTYPE *grid_uc;
};

/**
 * @brief The sync_manager struct manages multicast memory spaces across 
 * multiple GPUs for synchronization. It is just a named PGL under the hood.
 * 
 * If sync_level is BLOCK, you must set MAX_BLOCKS.
 * If sync_level is GRID, MAX_BLOCKS is ignored.
 */
template <int NUM_DEVICES, sync_level LEVEL, int MAX_SYNC_POINTS = 4, int MAX_BLOCKS = -1>
struct sync_manager {
    using identifier = ducks::sync_manager::identifier;

    static_assert(NUM_DEVICES > 0, "NUM_DEVICES must be greater than 0");
    static_assert(LEVEL == sync_level::BLOCK || LEVEL == sync_level::GRID, "LEVEL must be either BLOCK or GRID");
    static_assert(LEVEL == sync_level::GRID || MAX_BLOCKS > 0, "MAX_BLOCKS must be greater than 0 for BLOCK level sync");
    static_assert(MAX_SYNC_POINTS > 0, "MAX_SYNC_POINTS must be greater than 0");

    static constexpr sync_level level = LEVEL;
    static constexpr int sync_point_size = level == sync_level::GRID ? 2 : MAX_BLOCKS;
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
     * Get the sync point for a specific sync ID.
     * block_idx is only used for block level sync, ignored for grid-level sync.
     */
    __device__ inline sync_point<SYNC_SPACE_DTYPE, LEVEL> get_sync_point(const int sync_id, const int dev_idx, const int block_idx = -1) const {
        if constexpr (LEVEL == sync_level::GRID) {
            return sync_point<SYNC_SPACE_DTYPE, LEVEL>{
                sync_space.mc_vas[dev_idx] + sync_id * sync_point_size,
                sync_space[dev_idx].raw_ptr + sync_id * sync_point_size,
                sync_space[dev_idx].raw_ptr + sync_id * sync_point_size + 1,
            };
        } else if constexpr (LEVEL == sync_level::BLOCK) {
            return sync_point<SYNC_SPACE_DTYPE, LEVEL>{
                sync_space.mc_vas[dev_idx] + sync_id * sync_point_size + block_idx,
                sync_space[dev_idx].raw_ptr + sync_id * sync_point_size + block_idx,
            };
        } else {
            return sync_point<SYNC_SPACE_DTYPE, LEVEL>{}; // this will raise a compile-time error
        }
    }
};

} // namespace kittens
