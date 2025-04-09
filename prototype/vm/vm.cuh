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
    __shared__ alignas(128) int instructions[config::INSTRUCTION_PIPELINE_STAGES][config::INSTRUCTION_WIDTH];
    __shared__ alignas(128) int timings[config::INSTRUCTION_PIPELINE_STAGES][config::TIMING_WIDTH];
    __shared__ uint32_t page_assignment_counter[2]; // contiguous
    __shared__ int page_assignment[config::PAGE_RING_SIZE],
                   mini_page_assignment[config::PAGE_RING_SIZE];
    __shared__ kittens::semaphore page_arrived[config::NUM_PAGES],
                                  page_finished[config::NUM_PAGES],
                                  mini_page_arrived[config::NUM_MINI_PAGES],
                                  mini_page_finished[config::NUM_MINI_PAGES],
                                  instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],
                                  instruction_finished[config::INSTRUCTION_PIPELINE_STAGES];
    extern __shared__ int __shm[];
    void *aligned_shm_addr = (void*)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);
    typename state<config>::page_array_t &pages = *reinterpret_cast<typename state<config>::page_array_t*>(aligned_shm_addr);
    typename state<config>::mini_page_array_t &mini_pages = *reinterpret_cast<typename state<config>::mini_page_array_t*>((uint64_t)aligned_shm_addr + config::PAGE_SIZE*config::NUM_PAGES);
    uint32_t base_page_assignment_counter = static_cast<uint32_t>(__cvta_generic_to_shared(&page_assignment_counter[0]));
    typename state<config>::tensor_allocator_t tensor_alloc{};

#ifdef KVM_DEBUG
    if(threadIdx.x == 0) printf("Pre-KVMS creation\n"); group<config::NUM_WARPS>::sync(15);
#endif

    state<config> kvms {
        instructions,
        timings,
        instruction_arrived, instruction_finished,
        0, 0,
        pages,
        mini_pages,
        page_arrived, page_finished,
        mini_page_arrived, mini_page_finished,
        page_assignment, mini_page_assignment,
        0, 0,
        base_page_assignment_counter,
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
            timings[i][threadIdx.x] = 0;
        }
    }
    
    if(threadIdx.x < 2) {
        page_assignment_counter[threadIdx.x] = 0;
    }
    if(threadIdx.x < config::PAGE_RING_SIZE) {
        page_assignment[threadIdx.x] = 0;
        mini_page_assignment[threadIdx.x] = 0;
    }
    if(threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 0, 2); // One arrival for instruction arriving, one for timing writeout finishing.
        init_semaphore(instruction_finished[threadIdx.x], 0, config::NUM_WARPS);
    }
    if(threadIdx.x < config::NUM_PAGES) {
        init_semaphore(page_arrived[threadIdx.x], 0, config::NUM_CONSUMER_WARPS);
        init_semaphore(page_finished[threadIdx.x], 0, config::NUM_CONSUMER_WARPS);
    }
    if(threadIdx.x < config::NUM_MINI_PAGES) {
        init_semaphore(mini_page_arrived[threadIdx.x], 0, config::NUM_CONSUMER_WARPS);
        init_semaphore(mini_page_finished[threadIdx.x], 0, config::NUM_CONSUMER_WARPS);
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