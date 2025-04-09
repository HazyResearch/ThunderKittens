#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "config.cuh"

namespace kittens {
namespace prototype {
namespace vm {

template<typename config, typename globals, typename state_type, template<typename> typename op_dispatcher, typename... ops>
struct dispatch_op {
    __device__ static inline void run(int opcode, const globals &g, state_type &state) {} // do nothing, base case
};
template<typename config, typename globals, typename state_type, template<typename> typename op_dispatcher, typename op, typename... ops>
struct dispatch_op<config, globals, state_type, op_dispatcher, op, ops...> {
    __device__ static inline void run(int opcode, const globals &g, state_type &state) {
        if(opcode == op::opcode) op_dispatcher<op>::run(g, state);
        else dispatch_op<config, globals, state_type, op_dispatcher, ops...>::run(opcode, g, state);
    }
};

template<typename config> struct page {
    int data[config::PAGE_SIZE / sizeof(int)];
};
template<typename config> struct mini_page {
    int data[config::MINI_PAGE_SIZE / sizeof(int)];
};

template<typename config> struct state {
    using instruction_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::INSTRUCTION_WIDTH];
    using timing_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::TIMING_WIDTH];
    using instruction_semaphore_array_t = kittens::semaphore[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_array_t &all_instructions;
    timing_array_t &all_timings;
    instruction_semaphore_array_t &instruction_arrived, &instruction_finished;
    int instruction_index, instruction_ring;

    __device__ inline int (&instruction())[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring];
    }
    __device__ inline const int (&instruction() const)[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring];
    }
    __device__ inline int (&timing())[config::TIMING_WIDTH] {
        return all_timings[instruction_ring];
    }
    __device__ inline const int (&timing() const)[config::TIMING_WIDTH] {
        return all_timings[instruction_ring];
    }
    __device__ inline void await_instruction() const {
        wait(instruction_arrived[instruction_ring], (instruction_index/config::INSTRUCTION_PIPELINE_STAGES)&1);
    }
    __device__ inline void next_instruction() {
        __syncwarp();
        if(laneid() == 0) {
#ifdef KVM_DEBUG
            printf("Thread %d: arriving at instruction finished %d\n", threadIdx.x, instruction_ring);
#endif
            kittens::arrive(instruction_finished[instruction_ring]);
        }
        instruction_index++;
        instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(instruction_ring);
    }

    using page_array_t = page<config>[config::NUM_PAGES];
    using mini_page_array_t = mini_page<config>[config::NUM_MINI_PAGES];
    page_array_t &pages;
    mini_page_array_t &mini_pages;

    using page_semaphore_array_t = kittens::semaphore[config::NUM_PAGES];
    using mini_page_semaphore_array_t = kittens::semaphore[config::NUM_MINI_PAGES];
    page_semaphore_array_t &page_arrived, &page_finished;
    mini_page_semaphore_array_t &mini_page_arrived, &mini_page_finished;

    using page_assignment_array_t = int[config::PAGE_RING_SIZE];
    page_assignment_array_t &page_assignment, &mini_page_assignment;
    int page_iter, mini_page_iter; // Count page assignments, ring is merely (iter % config::PAGE_RING_SIZE)
    __device__ inline int page_ring() const { return page_iter % config::PAGE_RING_SIZE; }
    __device__ inline int mini_page_ring() const {return mini_page_iter % config::PAGE_RING_SIZE; }
    uint32_t _page_assignment_counter; // this is a shared memory address incremented as pages are assigned.
    __device__ inline uint32_t page_assignment_counter() const { return _page_assignment_counter; }
    __device__ inline uint32_t mini_page_assignment_counter() const { return _page_assignment_counter+sizeof(uint32_t); }

    template<int distance=1> __device__ inline int get_page() {
        page_iter += (distance-1);
        while(true) {
            int counter;
            move<int>::lds(counter, page_assignment_counter());
            if(counter >= page_iter) break;
            __nanosleep(20); // poll until the next page is assigned.
        }
        int next_page;
        next_page = page_assignment[page_ring()];
        page_iter++;
        return next_page;
    }
    template<int distance=1> __device__ inline int get_mini_page() {
        mini_page_iter += distance;
        while(true) {
            int counter;
            move<int>::lds(counter, mini_page_assignment_counter());
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
        timing()[event_id] = diff;
    }

    static constexpr int NCTA_TENSOR_ALLOC = config::CLUSTER_BLOCKS > 1 ? 2 : 1;
    using tensor_allocator_t = ::kittens::tensor_allocator<1, NCTA_TENSOR_ALLOC>;
    tensor_allocator_t &tensor_alloc;

    __device__ inline void print() {
        printf("Kittens Virtual Machine State being printed by thread %d, block %d\n", threadIdx.x, blockIdx.x);
        printf("Instruction index: %d, Instruction ring: %d\n", instruction_index, instruction_ring);
        printf("Page iter: %d, Mini page iter: %d\n", page_iter, mini_page_iter);    
    }
};

} // namespace vm
} // namespace prototype
} // namespace kittens

#ifdef KVM_DEBUG
#define KVM_DEBUG_PRINT(msg) printf("Thread %d: starting main loop for %s\n", threadIdx.x, msg);
#else
#define KVM_DEBUG_PRINT(msg)
#endif


#define MAKE_WORKER(name) \
namespace kittens { \
namespace prototype { \
namespace vm { \
namespace name { \
\
template<typename config, typename globals> \
struct name##_op_dispatcher { \
    template<typename op> \
    struct dispatcher { \
        __device__ static inline void run(const globals &g, ::kittens::prototype::vm::state<config> &kvms) { \
            op::name::run(g, kvms);    \
        } \
    }; \
}; \
\
template<typename config, typename globals, typename... ops> __device__ void main_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) { \
    KVM_DEBUG_PRINT(#name); \
    int num_iters = g.instructions.rows(); \
    for(kvms.instruction_index = 0, kvms.instruction_ring = 0; kvms.instruction_index < num_iters; kvms.next_instruction()) { \
        kvms.await_instruction(); \
        dispatch_op<config, globals, ::kittens::prototype::vm::state<config>, name##_op_dispatcher<config, globals>::dispatcher, ops...>::run(kvms.instruction()[0], g, kvms); \
    } \
} \
\
} \
} \
} \
}
