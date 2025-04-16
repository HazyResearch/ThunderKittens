#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"
#include "instruction_fetch.cuh"
#include "timings_store.cuh"
#include "page_allocator.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals, typename... ops> __device__ void main_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    int lane = kittens::laneid();
    if(lane < config::NUM_PAGES) {
#ifdef KVM_DEBUG
        printf("Thread %d: starting page allocator loop\n", threadIdx.x);
#endif
        page_allocator_loop<config, globals, ops...>(g, kvms);
    } else if(lane == 30) {
#ifdef KVM_DEBUG
        printf("Thread %d: starting timings store loop\n", threadIdx.x);
#endif
        timings_store_loop<config, globals>(g, kvms); // Doesn't need an ops list.
    } else if(lane == 31) {
#ifdef KVM_DEBUG
        printf("Thread %d: starting instruction fetch loop\n", threadIdx.x);
#endif
        instruction_fetch_loop<config, globals>(g, kvms); // Doesn't need an ops list.
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
