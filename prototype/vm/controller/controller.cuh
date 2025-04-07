#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"
#include "util.cuh"

#include "instruction_fetch.cuh"
#include "timings_store.cuh"
#include "page_allocator.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals, typename... ops> __global__ void main_loop(globals &g, state<config> &cs) {
    int lane = kittens::laneid();
    constexpr int MAX_PAGES = config::NUM_PAGES > config::NUM_MINI_PAGES ? config::NUM_PAGES : config::NUM_MINI_PAGES;
    if(lane < MAX_PAGES) {
        page_allocator_loop<config, globals, MAX_PAGES, ops...>(g, cs);
    } else if(lane == 30) {
        timings_store_loop<config, globals>(g, cs); // Doesn't need an ops list.
    } else if(lane == 31) {
        instruction_fetch_loop<config, globals>(g, cs); // Doesn't need an ops list.
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
