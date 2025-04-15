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
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;

    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;
    static constexpr int STATIC_SHARED_MEMORY = 3000;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static constexpr int MINI_PAGE_SIZE = 1024;
    static constexpr int NUM_MINI_PAGES = (DYNAMIC_SHARED_MEMORY % PAGE_SIZE) / MINI_PAGE_SIZE;
    static constexpr int PAGE_RING_SIZE = 32;

    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
};
template<typename config> using instruction_layout = gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;
template<typename config> using timing_layout      = gl<int, 1, -1, -1, config::TIMING_WIDTH>;
template<typename config> using semaphore_layout   = gl<int, 1, 1, -1, -1>;

template<typename config> void print_config() {
    std::cout << "---------------- CONFIG INFO ----------------" << std::endl;
    std::cout << "NUM_CONSUMER_WARPS: " << config::NUM_CONSUMER_WARPS << std::endl;
    std::cout << "NUM_WARPS: " << config::NUM_WARPS << std::endl;
    std::cout << "NUM_THREADS: " << config::NUM_THREADS << std::endl;
    std::cout << "NUM_BLOCKS: " << config::NUM_BLOCKS << std::endl;
    std::cout << "CLUSTER_BLOCKS: " << config::CLUSTER_BLOCKS << std::endl;
    std::cout << "MAX_SHARED_MEMORY: " << config::MAX_SHARED_MEMORY << std::endl;
    std::cout << "STATIC_SHARED_MEMORY: " << config::STATIC_SHARED_MEMORY << std::endl;
    std::cout << "PAGE_SIZE: " << config::PAGE_SIZE << std::endl;
    std::cout << "NUM_PAGES: " << config::NUM_PAGES << std::endl;
    std::cout << "MINI_PAGE_SIZE: " << config::MINI_PAGE_SIZE << std::endl;
    std::cout << "NUM_MINI_PAGES: " << config::NUM_MINI_PAGES << std::endl;
    std::cout << "PAGE_RING_SIZE: " << config::PAGE_RING_SIZE << std::endl;
    std::cout << "INSTRUCTION_PIPELINE_STAGES: " << config::INSTRUCTION_PIPELINE_STAGES << std::endl;
    std::cout << "INSTRUCTION_WIDTH: " << config::INSTRUCTION_WIDTH << std::endl;
    std::cout << "TIMING_WIDTH: " << config::TIMING_WIDTH << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

} // namespace vm
} // namespace prototype
} // namespace kittens
