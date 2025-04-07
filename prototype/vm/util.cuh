#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"

namespace kittens {
namespace prototype {
namespace vm {

template<typename config, typename globals, typename state_type, bool op_dispatcher, typename... ops>
struct dispatch_op {
    __device__ static inline void run(int opcode, const globals &g, state_type &state) {} // do nothing, base case
};
template<typename config, typename globals, typename state_type, bool op_dispatcher, typename op, typename... ops>
struct dispatch_op<config, globals, state_type, op_dispatcher, op, ops...> {
    __device__ static inline void run(int opcode, const globals &g, state_type &state) {
        if(opcode == op::opcode) op_dispatcher<config, globals, op>::run(g, state);
        else dispatch_op<config, globals, state_type, op_dispatcher, ops...>::run(opcode, g, state);
    }
};

template<typename config> struct vm_state {
    using instruction_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::INSTRUCTION_WIDTH];
    using timing_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::TIMING_EVENTS];
    using instruction_semaphore_array_t = kittens::semaphore[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_array_t &instructions;
    timing_array_t &timings;
    instruction_semaphore_array_t &instruction_arrived, &instruction_finished;
    int task_iter, instruction_ring;

    __device__ inline int (&instruction())[config::INSTRUCTION_WIDTH] {
        return instructions[instruction_ring];
    }
    __device__ inline const int (&instruction())[config::INSTRUCTION_WIDTH] const {
        return instructions[instruction_ring];
    }
    __device__ inline void next_instruction() {
        __syncwarp();
        kittens::warp::arrive(instruction_finished[instruction_ring]);
        task_iter++;
        instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(instruction_ring);
        wait(instruction_arrived[instruction_ring], (task_iter/config::INSTRUCTION_PIPELINE_STAGES)&1);
    }

    using page_semaphore_array_t = kittens::semaphore[config::NUM_PAGES];
    using mini_page_semaphore_array_t = kittens::semaphore[config::NUM_MINI_PAGES];

    page_semaphore_array_t &page_arrived, &page_finished;
    mini_page_semaphore_array_t &mini_page_arrived, &mini_page_finished;

    using page_assignment_array_t = int[config::PAGE_RING_SIZE];
    page_assignment_array_t &page_assignment, &mini_page_assignment;
    int page_iter, mini_page_iter; // Count page assignments, ring is merely (iter % config::PAGE_RING_SIZE)
    __device__ inline int page_ring() const { return (page_iter-1) % config::PAGE_RING_SIZE; }
    __device__ inline int mini_page_ring() const {return (mini_page_iter-1) % config::PAGE_RING_SIZE; }
    uint32_t page_assignment_counter, mini_page_assignment_counter; // this is a shared memory address incremented as pages are assigned.

    template<int distance=1> __device__ inline int get_page() {
        page_iter += distance;
        while(true) {
            int counter;
            move::lds<int>(counter, page_assignment_counter);
            if(counter >= page_iter) break;
            __nanosleep(20); // poll until the next page is assigned.
        }
        int next_page;
        next_page = page_assignment[page_ring()];
        return next_page;
    }
    template<int distance=1> __device__ inline int get_mini_page() {
        mini_page_iter += distance;
        while(true) {
            int counter;
            move::lds<int>(counter, mini_page_assignment_counter);
            if(counter >= mini_page_iter) break;
            __nanosleep(20); // poll until the next mini page is assigned.
        }
        int next_mini_page;
        next_mini_page = mini_page_assignment[mini_page_ring()];
        return next_mini_page;
    }
    __device__ inline void wait_page_arrived(int id) {
        wait(page_arrived[id], 1);
    }
    __device__ inline void wait_mini_page_arrived(int id) {
        wait(mini_page_arrived[id], 1);
    }

    uint64_t start_clock;
    
    __device__ inline void record(int event_id) {
        uint64_t current = clock64();
        int diff = (int)(current - start_clock);
        timings[instruction_ring][event_id] = diff;
    }
};

} // namespace vm
} // namespace prototype
} // namespace kittens