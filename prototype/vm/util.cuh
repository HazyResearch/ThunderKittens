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


__device__ inline unsigned int get_smid() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ inline unsigned int get_worker_id() {
    return get_smid();
    // return blockIdx.x;
}


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

    template<int num_bytes>
    __device__ inline void zero_scratch() {
        static_assert(num_bytes % 4 == 0, "num_bytes must be a multiple of 4");
        constexpr auto num_floats = num_bytes / 4;
        auto &scratch_vec = *reinterpret_cast<sv_fl<num_floats>*>(scratch());
        warp::zero(scratch_vec);
        warp::sync();
    }

    __device__ inline kittens::semaphore (&semaphores())[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline const kittens::semaphore (&semaphores() const)[config::DYNAMIC_SEMAPHORES] {
        return all_instructions[instruction_ring].semaphores;
    }
    __device__ inline void await_instruction() {
        wait(instruction_arrived[instruction_ring], (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1);
        pid_order_shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&(pid_order()[0])));
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

    using page_semaphore_array_t = kittens::semaphore[config::NUM_PAGES][config::INSTRUCTION_PIPELINE_STAGES_BITS];
    page_semaphore_array_t &page_finished;

    __device__ inline int pid(int lid) {
        int ret;
        move<int>::lds(ret, pid_order_shared_addr + lid*sizeof(int));
        return ret;
    }
    __device__ inline void wait_page_ready(int pid) {
        #pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            auto bit = (instruction_index >> i) & 1;
            wait(page_finished[pid][i], bit);
        }
    }

    __device__ inline void finish_page(int pid, int count) {
        #pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            arrive(page_finished[pid][i], count);
        }
    }

    __device__ inline void warp_finish_page(int pid, int count) {
        if (warp::laneid() == 0) {
            finish_page(pid, count);
        }
    }

#ifdef KITTENS_BLACKWELL
    semaphore &tensor_finished;
    __device__ inline void wait_tensor_ready() {
        wait(tensor_finished, instruction_index%2);
    }
#endif

    semaphore &semaphores_ready;
    __device__ inline void wait_semaphores_ready() {
        wait(semaphores_ready, instruction_index%2);
    }

    uint64_t start_clock;

    __device__ inline void record(int event_id) {
        if constexpr(config::TIMING_RECORD_ENABLED) {
            uint64_t current = clock64();
            int diff = (int)(current - start_clock);
            timing()[event_id] = diff;
        }
    }

#ifdef KITTENS_BLACKWELL
    static constexpr int NCTA_TENSOR_ALLOC = config::CLUSTER_BLOCKS > 1 ? 2 : 1;
    using tensor_allocator_t = ::kittens::tensor_allocator<1, NCTA_TENSOR_ALLOC>;
    tensor_allocator_t &tensor_alloc;
#endif

    uint32_t pid_order_shared_addr;

    __device__ inline void print() {
        printf("Kittens Virtual Machine State being printed by thread %d, block %d\n  Instruction index: %d, Instruction ring: %d\n", threadIdx.x, blockIdx.x, instruction_index, instruction_ring);
    }
};


// timing event convention
constexpr int TEVENT_CONTROLLER_START = 0;
constexpr int TEVENT_IFETCH_DONE = 1;
constexpr int TEVENT_PAGE_ALLOC_DONE = 2;
constexpr int TEVENT_SEMS_SETUP = 3;
constexpr int TEVENT_CONTROLLER_END = 4;
constexpr int TEVENT_LOADER_START = 5;
constexpr int TEVENT_LAUNCHER_START = 7;
constexpr int TEVENT_STORER_START = 9;

constexpr int TEVENT_CONSUMER_START = 11;

constexpr int TEVENT_AT_GMEM_WAIT = 44;
constexpr int TEVENT_DONE_GMEM_WAIT = 45;
constexpr int TEVENT_AT_GMEM_STORE = 46;
constexpr int TEVENT_DONE_GMEM_STORE = 47;

constexpr int TEVENT_FIRST_LOAD = 48;
constexpr int TEVENT_FIRST_USE = 49;
constexpr int TEVENT_FIRST_STORE = 50;

constexpr int TEVENT_LAST_LOAD = 51;
constexpr int TEVENT_LAST_USE = 52;
constexpr int TEVENT_LAST_STORE = 53;

constexpr int TEVENT_OUTPUT_READY = 54;

constexpr int FREE_SLOTS_START = 55;

constexpr int TEVENT_TRIPLES_START = 100;
constexpr int TEVENT_TRIPLES_END = 110;
constexpr int TEVENT_TRIPLES_STORE_START = 124;
constexpr int TEVENT_TRIPLES_OUTPUT_READY = 125;

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


#define MAKE_WORKER(name, start_event, is_consumer) \
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
        if (laneid() == 0) { \
            if (is_consumer) { \
                kvms.record(start_event + 2 * warpid()); \
            } \
            else { \
                kvms.record(start_event); \
            } \
        } \
        dispatch_op<name##_op_dispatcher<config, globals>::dispatcher, ops...>::template run<void, config, globals, ::kittens::prototype::vm::state<config>>(kvms.instruction()[0], g, kvms); \
        if (laneid() == 0) { \
            if (is_consumer) { \
                kvms.record(start_event + 2 * warpid() + 1); \
            } \
            else { \
                kvms.record(start_event + 1); \
            } \
        } \
    } \
    __syncwarp(); \
    KVM_DEBUG_PRINT_END(#name); \
} \
\
} \
} \
} \
}
