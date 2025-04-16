#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"
#include "config.cuh"
#include "util.cuh"
#include "controller/controller.cuh"
#include "launcher.cuh"
#include "storer.cuh"
#include "loader.cuh"
#include "consumer.cuh"

namespace kittens {
namespace prototype {
namespace vm {

template<typename config, typename globals, typename... ops>
__launch_bounds__(config::NUM_THREADS, 1)
__cluster_dims__(config::CLUSTER_BLOCKS)
__global__ void kernel(const __grid_constant__ globals g) {
#ifdef KVM_DEBUG
    if(threadIdx.x == 0) printf("Kernel launched\n"); group<config::NUM_WARPS>::sync(15);
#endif
    __shared__ alignas(128) instruction_state_t<config> instruction_state[config::INSTRUCTION_PIPELINE_STAGES];
    __shared__ kittens::semaphore page_finished[config::NUM_PAGES],
                                  instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],
                                  instruction_finished[config::INSTRUCTION_PIPELINE_STAGES];
    extern __shared__ int __shm[];
    void *aligned_shm_addr = (void*)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);
    typename state<config>::page_array_t &pages = *reinterpret_cast<typename state<config>::page_array_t*>(aligned_shm_addr);
    typename state<config>::tensor_allocator_t tensor_alloc{};

#ifdef KVM_DEBUG
    if(threadIdx.x == 0) printf("Pre-KVMS creation\n"); group<config::NUM_WARPS>::sync(15);
#endif

    state<config> kvms {
        instruction_state,
        instruction_arrived, instruction_finished,
        0, 0,
        { /* ... */ },
        pages,
        page_finished,
        (uint64_t)clock64(),
        tensor_alloc
    }; // kittens virtual machine state

#ifdef KVM_DEBUG
    if(threadIdx.x == 0) printf("Created KVMS\n"); group<config::NUM_WARPS>::sync(15);
#endif

    // Zero initial timings memory.
    if(threadIdx.x < config::TIMING_WIDTH) {
        #pragma unroll
        for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
            instruction_state[i].timings[threadIdx.x] = 0;
        }
    }
    
    if(threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 3); // One arrival for instruction arriving, one for timing writeout finishing.
        init_semaphore(instruction_finished[threadIdx.x], config::NUM_WARPS-1); // All but the controller warp arrive here.
    }
    if(threadIdx.x < config::NUM_PAGES) {
        init_semaphore(page_finished[threadIdx.x], config::NUM_CONSUMER_WARPS);
    }

    if(config::CLUSTER_BLOCKS == 1) group<config::NUM_WARPS>::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
    else everyone::tma::cluster::sync();

#ifdef KVM_DEBUG
    if(blockIdx.x == 0 && threadIdx.x == 0) kvms.print();
#endif

    if(warpid() < config::NUM_CONSUMER_WARPS) {
        warpgroup::increase_registers<104>();
        ::kittens::prototype::vm::consumer::main_loop<config, globals, ops...>(g, kvms);
    }
    else {
        warpgroup::decrease_registers<64>();
        switch(warpgroup::warpid()) {
            case 0:
                ::kittens::prototype::vm::loader::main_loop<config, globals, ops...>(g, kvms);
                break;
            case 1:
                ::kittens::prototype::vm::storer::main_loop<config, globals, ops...>(g, kvms);
                break;
            case 2:
                ::kittens::prototype::vm::launcher::main_loop<config, globals, ops...>(g, kvms);
                break;
            case 3:
                ::kittens::prototype::vm::controller::main_loop<config, globals, ops...>(g, kvms);
                break;
            default:
                asm volatile("trap;");
        }
    }

#ifdef KVM_DEBUG
    printf("Thread %d arriving at final barrier\n", threadIdx.x);
#endif

    if(config::CLUSTER_BLOCKS > 1) everyone::tma::cluster::sync();
    else group<config::NUM_WARPS>::sync(15);
}


} // namespace vm
} // namespace prototype
} // namespace kittens