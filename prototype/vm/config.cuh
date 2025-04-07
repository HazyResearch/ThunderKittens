#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {

struct default_config {
    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;

    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;
    static constexpr int STATIC_SHARED_MEMORY = 3000;
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = (MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY) / PAGE_SIZE;
    static constexpr int MINI_PAGE_SIZE = 256;
    static constexpr int NUM_MINI_PAGES = ((MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY) % PAGE_SIZE) / MINI_PAGE_SIZE;
    static constexpr int PAGE_RING_SIZE = 16;

    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2; // This is currently hardcoded, do not change.
    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.

    // Timing info
    static constexpr int TIMING_EVENTS = 128;
};
template<typename config> using instruction_layout = gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;
template<typename config> using timing_layout      = gl<int, 1, -1, -1, config::TIMING_EVENTS>;
template<typename config> using semaphore_layout   = gl<int, 1, 1, -1, -1>;

} // namespace vm
} // namespace prototype
} // namespace kittens
