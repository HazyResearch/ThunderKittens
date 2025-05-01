#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "config.cuh"

namespace kittens {
namespace prototype {
namespace vm {


// pid -- physical page id
// lid -- logical page id

template<typename config> struct __align__(128) instruction_state_t {
    config::instruction_t instructions;
    config::timing_t timings;
    int pid_order[config::NUM_PAGES];
    int padding[((config::NUM_PAGES + 31) & ~31) - config::NUM_PAGES]; // Round up to multiple of 32
    kittens::semaphore semaphores[config::DYNAMIC_SEMAPHORES];
    int scratch[config::SCRATCH_BYTES/4];
};


template<template<typename> typename op_dispatcher, typename... ops>
struct dispatch_op {
    template<typename return_t, typename config, typename globals, typename... args>
    __device__ static inline return_t run(int opcode, const globals &g, args&... a) {
        asm volatile("trap;\n"); // we want to blow up in this case.
        return return_t{};
    } // do nothing, base case
};
template<template<typename> typename op_dispatcher, typename op, typename... ops>
struct dispatch_op<op_dispatcher, op, ops...> {
    template<typename return_t, typename config, typename globals, typename... args>
    __device__ static inline return_t run(int opcode, const globals &g, args&... a) {
        if(opcode == op::opcode) return op_dispatcher<op>::run(g, a...);
        else return dispatch_op<op_dispatcher, ops...>::template run<return_t, config, globals, args...>(opcode, g, a...);
    }
};

template<typename config> struct page {
    int data[config::PAGE_SIZE / sizeof(int)];
    template<typename T=fp8e4m3> __device__ inline auto &as_st() {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>, "Unsupported dtype for automatic cast. Please run your own reinterpret_cast!");
        if constexpr(std::is_same_v<T, fp8e4m3>) {
            return *reinterpret_cast<st_fp8e4m3<128, 128>*>(data);
        }
        else if constexpr(std::is_same_v<T, fp8e5m2>) {
            return *reinterpret_cast<st_fp8e5m2<128, 128>*>(data);
        }
        else if constexpr(std::is_same_v<T, float>) {
            return *reinterpret_cast<st_fl<64, 64>*>(data);
        }
    }
    __device__ inline void       *ptr(int byte_offset=0)       { return       (void*)(data + byte_offset / sizeof(int)); }
    __device__ inline const void *ptr(int byte_offset=0) const { return (const void*)(data + byte_offset / sizeof(int)); }
};
template<typename config> struct mini_page {
    int data[config::MINI_PAGE_SIZE / sizeof(int)];
};

template<typename config> struct state {
    using instruction_state_array_t = instruction_state_t<config>[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_state_array_t &all_instructions;
    using instruction_semaphore_array_t = kittens::semaphore[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_semaphore_array_t &instruction_arrived, &instruction_finished;
    int instruction_index, instruction_ring;
    int reg_pid_order[config::NUM_PAGES];

    __device__ inline int (&instruction())[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring].instructions;
    }
    __device__ inline const int (&instruction() const)[config::INSTRUCTION_WIDTH] {
        return all_instructions[instruction_ring].instructions;
    }
    __device__ inline int (&timing())[config::TIMING_WIDTH] {
        return all_instructions[instruction_ring].timings;
    }
    __device__ inline const int (&timing() const)[config::TIMING_WIDTH] {
        return all_instructions[instruction_ring].timings;
    }
    __device__ inline int (&pid_order())[config::NUM_PAGES] {
        return all_instructions[instruction_ring].pid_order;
    }
    __device__ inline const int (&pid_order() const)[config::NUM_PAGES] {
        return all_instructions[instruction_ring].pid_order;
    }
    __device__ inline void* scratch() const {
        return (void*)&all_instructions[instruction_ring].scratch[0];
    }
    __device__ inline kittens::semaphore (&semaphores())[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline const kittens::semaphore (&semaphores() const)[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline void await_instruction() {
        wait(instruction_arrived[instruction_ring], (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1);
        #pragma unroll
        for(int i = 0; i < config::NUM_PAGES; i++) {
            reg_pid_order[i] = pid_order()[i];
        }
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
    page_array_t &pages;

    using page_semaphore_array_t = kittens::semaphore[config::NUM_PAGES];
    page_semaphore_array_t &page_finished;

    __device__ inline int pid(int lid) {
        return reg_pid_order[lid];
    }
    __device__ inline void wait_page_ready(int pid) {
        wait(page_finished[pid], instruction_index%2);
    }

    semaphore &tensor_finished;
    __device__ inline void wait_tensor_ready() {
        wait(tensor_finished, instruction_index%2);
    }

    semaphore &semaphores_ready;
    __device__ inline void wait_semaphores_ready() {
        wait(semaphores_ready, instruction_index%2);
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
        printf("Kittens Virtual Machine State being printed by thread %d, block %d\n  Instruction index: %d, Instruction ring: %d\n", threadIdx.x, blockIdx.x, instruction_index, instruction_ring);
    }
};

} // namespace vm
} // namespace prototype
} // namespace kittens

#ifdef KVM_DEBUG
#define KVM_DEBUG_PRINT_START(msg) printf("Thread %d: starting main loop for %s\n", threadIdx.x, msg);
#define KVM_DEBUG_PRINT_END(msg) printf("Thread %d: exiting main loop for %s\n", threadIdx.x, msg);
#else
#define KVM_DEBUG_PRINT_START(msg)
#define KVM_DEBUG_PRINT_END(msg)
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
    KVM_DEBUG_PRINT_START(#name); \
    int num_iters = g.instructions.rows(); \
    for(kvms.instruction_index = 0, kvms.instruction_ring = 0; kvms.instruction_index < num_iters; kvms.next_instruction()) { \
        kvms.await_instruction(); \
        dispatch_op<name##_op_dispatcher<config, globals>::dispatcher, ops...>::template run<void, config, globals, ::kittens::prototype::vm::state<config>>(kvms.instruction()[0], g, kvms); \
    } \
    __syncwarp(); \
    KVM_DEBUG_PRINT_END(#name); \
} \
\
} \
} \
} \
}
