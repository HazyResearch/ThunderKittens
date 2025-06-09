#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <unordered_map>
#include <memory>
#include "detail/helpers.dp.hpp"
#include "../../common/common.dp.hpp"

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

template <typename SYNC_SPACE_DTYPE>
struct sync_point {
    SYNC_SPACE_DTYPE *sys_mc;
    SYNC_SPACE_DTYPE *sys_uc;
    SYNC_SPACE_DTYPE *dev_uc = nullptr;
};

/**
 * @brief The sync_manager struct manages multicast memory spaces across 
 * multiple GPUs for synchronization. It is just a named PGL under the hood.
 * 
 * This struct is meant to be passed to functions under the gang::sync namespace.
 * You can pass it to gang::sync::all(...) by default
 * You can pass it to gang::sync::blockwise(...) if MAX_BLOCKS > 0
 * You can pass it to gang::sync::blockgroup(...) if MAX_SYNC_POINTS > 0
 */
template <int NUM_DEVICES, int MAX_BLOCKS = 256, int MAX_SYNC_POINTS = 16>
struct sync_manager {
    static_assert(NUM_DEVICES > 0, "NUM_DEVICES must be greater than 0");
    static_assert(MAX_BLOCKS >= 0, "MAX_BLOCKS must be greater than or equal to 0");
    static_assert(MAX_SYNC_POINTS >= 0, "MAX_SYNC_POINTS must be greater than or equal to 0");

    using identifier = ducks::sync_manager::identifier;
    static constexpr int num_devices = NUM_DEVICES;
    static constexpr int all_sync_point_size = 2;
    static constexpr int max_blocks = MAX_BLOCKS;
    static constexpr int max_sync_points = MAX_SYNC_POINTS;
    static constexpr int sync_space_size = all_sync_point_size /* gang::everyone::sync()   */ + 
                                           MAX_BLOCKS          /* gang::blockwise::sync()  */ + 
                                           MAX_SYNC_POINTS     /* gang::blockgroup::sync() */;

    using SYNC_SPACE_DTYPE = int;
    using SYNC_SPACE_T = pgl<gl<SYNC_SPACE_DTYPE, 1, 1, 1, sync_space_size>, NUM_DEVICES, true>;
    SYNC_SPACE_T sync_space;

    // We recommend calling sync_manager.create() instead of direct construction
    inline sync_manager(int *device_ids, SYNC_SPACE_DTYPE **d_sync_spaces) : 
        sync_space(device_ids, d_sync_spaces, nullptr, nullptr, nullptr, nullptr) { }

    static inline sync_manager create(int* device_ids) {
        SYNC_SPACE_DTYPE *d_sync_spaces[NUM_DEVICES]; 

        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_id = device_ids[i];
            /*
            DPCT1093:474: The "dev_id" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(dev_id);
            pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_id, &d_sync_spaces[i], sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }

        return sync_manager(device_ids, d_sync_spaces);
    }

    inline void free() {
        for (int i = 0; i < NUM_DEVICES; ++i) {
            int dev_id = sync_space.device_ids[i];
            /*
            DPCT1093:475: The "dev_id" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(dev_id);
            pglCudaFree(dev_id, sync_space[i].raw_ptr, sync_space_size * sizeof(SYNC_SPACE_DTYPE));
        }
        pglFree(sync_space);
    }

    // For gang::everyone::sync()
    inline auto get_all_sync_point(const int dev_idx) const {
        return sync_point<SYNC_SPACE_DTYPE>{
            sync_space.mc_vas[dev_idx],
            sync_space[dev_idx].raw_ptr,
            sync_space[dev_idx].raw_ptr + 1
        };
    }

    // For gang::blockwise::sync()
    inline auto get_blockwise_sync_point(const int dev_idx, const int block_idx) const {
        return sync_point<SYNC_SPACE_DTYPE>{
            sync_space.mc_vas[dev_idx] + all_sync_point_size + block_idx,
            sync_space[dev_idx].raw_ptr + all_sync_point_size + block_idx,
            nullptr // unused
        };
    }

    // For gang::blockgroup::sync()
    inline auto get_blockgroup_sync_point(const int dev_idx, const int sync_id) const {
        return sync_point<SYNC_SPACE_DTYPE>{
            sync_space.mc_vas[dev_idx] + all_sync_point_size + max_blocks + sync_id,
            sync_space[dev_idx].raw_ptr + all_sync_point_size + max_blocks + sync_id,
            nullptr // unused
        };
    }
};

} // namespace kittens
