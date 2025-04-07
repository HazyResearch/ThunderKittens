#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {

struct page_alloc {
    int start, num;
};
struct base_controller_state {
    page_alloc pages;
    page_alloc mini_pages;
};
template<typename T> concept is_base_controller_state = requires(T t) {
    { t.pages.start } -> std::convertible_to<int>;
    { t.pages.num } -> std::convertible_to<int>;
    { t.mini_pages.start } -> std::convertible_to<int>;
    { t.mini_pages.num } -> std::convertible_to<int>;
};

struct page_state {
    void *addr;
    int index;
    semaphore *semaphore;
};
template<typename config> struct kittens_virtual_machine_state {
    int *instruction;
    int instruction_index;

    page_state pages;
    page_state mini_pages;

    __device__ inline page_alloc get_pages(int size_bytes) {
        int pages_needed = (size_bytes + config::PAGE_SIZE - 1) / config::PAGE_SIZE;
        if(pages.index + pages_needed > config::NUM_PAGES) pages.index = 0;
        int ret = pages.index;
        pages.index += pages_needed;
        return {ret, pages_needed};
    }
    __device__ inline page_alloc get_mini_pages(int size_bytes) {
        int mini_pages_needed = (size_bytes + config::MINI_PAGE_SIZE - 1) / config::MINI_PAGE_SIZE;
        if(mini_pages.index + mini_pages_needed > config::NUM_MINI_PAGES) mini_pages.index = 0;
        int ret = mini_pages.index;
        mini_pages.index += mini_pages_needed;
        return {ret, mini_pages_needed};
    }
    template<typename T=void> __device__ inline T *page_addr(int index) {
        return (T*)((uint64_t)(pages.addr) + index*config::PAGE_SIZE);
    }
    template<typename T=void> __device__ inline T *mini_page_addr(int index) {
        return (T*)((uint64_t)(mini_pages.addr) + index*config::MINI_PAGE_SIZE);
    }

    semaphore *global_semaphore_ready;
    int *global_semaphore_writeout;

#ifdef KITTENS_TIMINGS
    uint64_t start_t;
    int *timings;
    __device__ inline void record(int id) {
        timings[id] = (int)(clock64() - start_t);
    }
#endif
};
template<typename config> concept is_kvms = requires(kittens_virtual_machine_state<config> kvms) {
    // this doesn't need to be comprehensive, it just needs to filter out other objects
    { kvms.pages.addr } -> std::convertible_to<void*>;
    { kvms.pages.index } -> std::convertible_to<int>;
    { kvms.pages.semaphore } -> std::convertible_to<semaphore*>;
    { kvms.mini_pages.addr } -> std::convertible_to<void*>;
    { kvms.mini_pages.index } -> std::convertible_to<int>;
    { kvms.mini_pages.semaphore } -> std::convertible_to<semaphore*>;
    { kvms.instruction } -> std::convertible_to<int*>;
    { kvms.instruction_index } -> std::convertible_to<int>;
};

template<typename config, typename globals, bool is_producer, typename op> __device__ inline void run_op(const globals &g, kittens_virtual_machine_state<config> &kvms) {
    typename op::layout::controller_state control;
    if constexpr(is_producer) {
        typename op::layout::producer_state state;
        op::controller::setup(g, kvms, control);
        op::producer::setup(g, kvms, control, state);
        while(op::controller::run(g, kvms, control)) {
            op::producer::run(g, kvms, control, state);
            op::controller::advance(g, kvms, control);
        }
        op::controller::finish(g, kvms, control);
        op::producer::finish(g, kvms, control, state);
    }
    else {
        typename op::layout::consumer_state state;
        op::controller::setup(g, kvms, control);
        op::consumer::setup(g, kvms, control, state);
        while(op::controller::run(g, kvms, control)) {
            op::consumer::run(g, kvms, control, state);
            op::controller::advance(g, kvms, control);
        }
        op::controller::finish(g, kvms, control);
        op::consumer::finish(g, kvms, control, state);
    }
    __syncwarp();
}



template<is_kvms KVMS, is_base_controller_state controller_state> __device__ inline void wait(KVMS &kvms, const controller_state &control, int phase) {
    #pragma unroll
    for(int i = 0; i < control.pages.num; i++) {
        kittens::wait(kvms.pages.semaphore[control.pages.start+i], phase);
    }
    #pragma unroll
    for(int i = 0; i < control.mini_pages.num; i++) {
        kittens::wait(kvms.mini_pages.semaphore[control.mini_pages.start+i], phase);
    }
}
template<is_kvms KVMS, is_base_controller_state controller_state> __device__ inline void wait_arrived(KVMS &kvms, const controller_state &control) {
    wait<KVMS, controller_state>(kvms, control, 0);
}
template<is_kvms KVMS, is_base_controller_state controller_state> __device__ inline void wait_finished(KVMS &kvms, const controller_state &control) {
    wait<KVMS, controller_state>(kvms, control, 1);
}
template<is_kvms KVMS, is_base_controller_state controller_state> __device__ inline void arrive(KVMS &kvms, const controller_state &control, int count=1) {
    if(laneid() == 0) {
        for(int i = 0; i < control.pages.num; i++) {
            kittens::arrive(kvms.pages.semaphore[control.pages.start+i], count);
        }
        for(int i = 0; i < control.mini_pages.num; i++) {
            kittens::arrive(kvms.mini_pages.semaphore[control.mini_pages.start+i], count);
        }
    }
}
template<is_kvms KVMS, is_base_controller_state controller_state, typename... expectation_types> __device__ inline kittens::semaphore &expect(KVMS &kvms, const controller_state &control, int count, expectation_types... expectations) {
    if(control.pages.num > 0) {
        tma::expect(kvms.pages.semaphore[control.pages.start], expectations...);
        if(laneid() == 0) {
            if(count > 1) kittens::arrive(kvms.pages.semaphore[control.pages.start], count-1);
            for(int i = 1; i < control.pages.num; i++) {
                kittens::arrive(kvms.pages.semaphore[control.pages.start+i], count);
            }
            for(int i = 0; i < control.mini_pages.num; i++) {
                kittens::arrive(kvms.mini_pages.semaphore[control.mini_pages.start+i], count);
            }
        }
        return kvms.pages.semaphore[control.pages.start];
    }
    else if(control.mini_pages.num > 0) {
        tma::expect(kvms.mini_pages.semaphore[control.mini_pages.start], expectations...);
        if(laneid() == 0) {
            if(count > 1) kittens::arrive(kvms.mini_pages.semaphore[control.mini_pages.start], count-1);
            for(int i = 0; i < control.mini_pages.num; i++) {
                kittens::arrive(kvms.mini_pages.semaphore[control.mini_pages.start+i], count);
            }
        }
        return kvms.mini_pages.semaphore[control.mini_pages.start];
    }
    else asm volatile("trap;\n"); // bad
    return kvms.pages.semaphore[0]; // never reached, but silences compiler warnings.
}
template<is_kvms KVMS, is_base_controller_state controller_state, typename... expectation_types> __device__ inline kittens::semaphore &expect(KVMS &kvms, const controller_state &control, expectation_types... expectations) {
    return expect<KVMS, controller_state>(kvms, control, 1, expectations...);
}

template<typename config, typename globals, typename... ops>
__launch_bounds__((config::NUM_CONSUMER_WARPS+config::NUM_PRODUCER_WARPS)*WARP_THREADS, 1)
__cluster_dims__(config::CLUSTER_BLOCKS)
__global__ void kernel(const __grid_constant__ globals g) {
    kittens_virtual_machine_state<config> kvms;
#ifdef KITTENS_TIMINGS
    kvms.start_t = clock64();
    __shared__ __align__(128) int timings[config::INSTRUCTION_PIPELINE_STAGES][config::TIMING_EVENTS]; // We'll allow 64 separate timing events, per instruction.
#endif
    // Zero semaphore to global memory.
    __shared__ int global_semaphore_writeout_buffer[config::INSTRUCTION_PIPELINE_STAGES];
    if(threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) global_semaphore_writeout_buffer[threadIdx.x] = 1; // Default to 1.
    for(int i = threadIdx.x; i < g.instructions.rows(); i+=blockDim.x) {
        // Volatile store of 0 to semaphore at the current block and instruction index
        *(volatile int*)&g.semaphore[kittens::coord<>{int(blockIdx.x), i}] = 0;
    }
    __shared__ __align__(128) int instructions[config::INSTRUCTION_PIPELINE_STAGES][config::INSTRUCTION_WIDTH];
    __shared__ kittens::semaphore page_semaphore[config::NUM_PAGES],
                                  mini_page_semaphore[config::NUM_MINI_PAGES],
                                  instruction_semaphore[config::INSTRUCTION_PIPELINE_STAGES],
                                  global_semaphore_ready[config::INSTRUCTION_PIPELINE_STAGES];
    if(warpid() == 0) {
        for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
            init_semaphore(instruction_semaphore[i], 0, config::NUM_WARPS);
        }
        load_instructions<config, globals>(&instructions[0][0], 0, g, instruction_semaphore[0]);
#ifdef KITTENS_TIMINGS
        for(int timing_id = laneid(); timing_id < config::TIMING_EVENTS; timing_id += WARP_THREADS) { // 0 them all to start.
            #pragma unroll
            for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
                timings[i][timing_id] = 0;
            }
        }
        __syncwarp();
#endif
        for(int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
            if(laneid() == 0) kittens::arrive(instruction_semaphore[i], config::NUM_WARPS-1);
        }
    }
    for(int i = warpid(); i < config::NUM_PAGES; i += config::NUM_WARPS) {
        init_semaphore(page_semaphore[i], 0, 32); // They're used for both I/O so it's sort of easier to just have a fixed number.
    }
    for(int i = warpid(); i < config::NUM_MINI_PAGES; i += config::NUM_WARPS) {
        init_semaphore(mini_page_semaphore[i], 0, 32); // Ditto as above.
    }
    for(int i = warpid(); i < config::INSTRUCTION_PIPELINE_STAGES; i += config::NUM_WARPS) {
        init_semaphore(global_semaphore_ready[i], 0, config::NUM_WARPS);
    }
    extern __shared__ int __shm[];
    void *aligned_shm_addr = (void*)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);
    kvms.pages.addr = aligned_shm_addr;
    kvms.pages.index = 0;
    kvms.pages.semaphore = &page_semaphore[0];
    kvms.mini_pages.addr = (void*)((uint64_t)(aligned_shm_addr) + config::PAGE_SIZE*config::NUM_PAGES);
    kvms.mini_pages.index = 0;
    kvms.mini_pages.semaphore = &mini_page_semaphore[0];
    
    if(config::CLUSTER_BLOCKS == 1) group<config::NUM_WARPS>::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
    else tma::cluster::sync();
    // grid::sync();
    group<config::NUM_WARPS>::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.

    if(warpid() < config::NUM_CONSUMER_WARPS) {
        warpgroup::increase_registers<224>();
        // if(laneid() == 0) printf("%d %d Consumer started\n", blockIdx.x, threadIdx.x); __syncwarp();
        for(kvms.instruction_index = 0; kvms.instruction_index < g.instructions.rows(); kvms.instruction_index++) {
            int instruction_stage = kvms.instruction_index%config::INSTRUCTION_PIPELINE_STAGES;
            kvms.instruction = &instructions[instruction_stage][0];
            kvms.global_semaphore_ready = &global_semaphore_ready[instruction_stage];
            kvms.global_semaphore_writeout = &global_semaphore_writeout_buffer[instruction_stage];
#ifdef KITTENS_TIMINGS
            kvms.timings = &timings[instruction_stage][0];
#endif
            kittens::wait(instruction_semaphore[instruction_stage], 0);
            // if(laneid() == 0) printf("warp %d finished waiting for instruction semaphore %d\n", warpid(), kvms.instruction_index); __syncwarp();
            int opcode = kvms.instruction[0];
            if(opcode == 0) break; // Stop Op
            // if(laneid() == 0) printf("%d %d Consumer started 0\n", blockIdx.x, threadIdx.x); __syncwarp();
            dispatch_ops<config, globals, false, ops...>::run(opcode, g, kvms);
            // if(laneid() == 0) printf("warp %d finished dispatching\n", warpid()); __syncwarp();
            // if(laneid() == 0) printf("%d %d Consumer started 1\n", blockIdx.x, threadIdx.x); __syncwarp();
            if(laneid() == 0) kittens::arrive(instruction_semaphore[instruction_stage]);
            // if(laneid() == 0) printf("%d %d Consumer started 2\n", blockIdx.x, threadIdx.x); __syncwarp();
        }
        // if(laneid() == 0) printf("%d %d Consumer finished\n", blockIdx.x, threadIdx.x); __syncwarp();
    }
    else {
        // if(laneid() == 0) printf("%d %d Producer began\n", blockIdx.x, threadIdx.x); __syncwarp();
        warpgroup::decrease_registers<40>();
        for(kvms.instruction_index = 0; kvms.instruction_index < g.instructions.rows(); kvms.instruction_index++) {
            if(warpgroup::warpid() == 3 && kvms.instruction_index+1 < g.instructions.rows()) {
                int next_instruction_stage = (kvms.instruction_index+1)%config::INSTRUCTION_PIPELINE_STAGES;
                kittens::wait(instruction_semaphore[next_instruction_stage], 1);
                load_instructions<config, globals>(&instructions[next_instruction_stage][0], 
                                                kvms.instruction_index+1, g, instruction_semaphore[next_instruction_stage]);
            }
            int instruction_stage = kvms.instruction_index%config::INSTRUCTION_PIPELINE_STAGES;
            kvms.instruction = &instructions[instruction_stage][0];
            kvms.global_semaphore_ready = &global_semaphore_ready[instruction_stage];
            kvms.global_semaphore_writeout = &global_semaphore_writeout_buffer[instruction_stage];
#ifdef KITTENS_TIMINGS
            kvms.timings = &timings[instruction_stage][0];
#endif
            kittens::wait(instruction_semaphore[instruction_stage], 0);
            // if(laneid() == 0) printf("warp %d finished waiting for instruction semaphore %d\n", warpid(), kvms.instruction_index); __syncwarp();
            int opcode = kvms.instruction[0];
            if(opcode == 0) break; // Stop Op
            dispatch_ops<config, globals, true, ops...>::run(opcode, g, kvms);
            // if(laneid() == 0) printf("warp %d finished dispatching\n", warpid()); __syncwarp();
            // if(laneid() == 0) printf("%d %d Producer 0\n", blockIdx.x, threadIdx.x); __syncwarp();
            if(laneid() == 0) kittens::arrive(instruction_semaphore[instruction_stage]);
            // if(laneid() == 0) printf("%d %d Producer 1\n", blockIdx.x, threadIdx.x); __syncwarp();
            if(warpgroup::warpid() == 3) {
                // if(laneid() == 0) printf("%d %d waiting for global semaphore ready\n", blockIdx.x, threadIdx.x); __syncwarp();
                kittens::wait(global_semaphore_ready[instruction_stage], 0);
                // if(laneid() == 0) printf("%d %d finished waiting for global semaphore ready\n", blockIdx.x, threadIdx.x); __syncwarp();
                if(laneid() == 0) {
                    kittens::arrive(global_semaphore_ready[instruction_stage], config::NUM_WARPS); // Send it back to phase 1.
                    *(volatile int*)&g.semaphore[kittens::coord<>{(int)(blockIdx.x), kvms.instruction_index}] = global_semaphore_writeout_buffer[instruction_stage];
                    global_semaphore_writeout_buffer[instruction_stage] = 1;
                }
#ifdef KITTENS_TIMINGS
                // if(laneid() == 0) printf("waiting for instruction semaphore\n"); __syncwarp();
                kittens::wait(instruction_semaphore[instruction_stage], 1);
                // if(laneid() == 0) printf("finished waiting for instruction semaphore\n"); __syncwarp();
                write_timings<config, globals>(kvms.timings, kvms.instruction_index, g);
                tma::store_async_read_wait();
                __syncwarp();
                for(int i = laneid(); i < config::TIMING_EVENTS; i += WARP_THREADS) {
                    kvms.timings[i] = 0;
                }
                __syncwarp();
#endif
                // if(laneid() == 0) printf("arriving at instruction semaphore\n"); __syncwarp();
                if(laneid() == 0) kittens::arrive(instruction_semaphore[instruction_stage], config::NUM_WARPS-1);
                // if(laneid() == 0) printf("finished arriving at instruction semaphore\n"); __syncwarp();
            }
        }
        // if(laneid() == 0) printf("%d %d Producer finished\n", blockIdx.x, threadIdx.x); __syncwarp();
    }

    if(config::CLUSTER_BLOCKS > 1) tma::cluster::sync();

}



} // namespace vm
} // namespace prototype
} // namespace kittens