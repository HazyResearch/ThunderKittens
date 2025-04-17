#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

/*
Instruction format:
[0] = opcode
[1] = Row offset of C, in units of 128
[2] = Col offset of C, in units of 128
[3] = K reduction dimension, in units of 128
*/

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, st_fp8e4m3<128, 128>>;
    instruction_layout instructions;
    timing_layout timings;
    fp8_matrix A, B, C;
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 1;
    static constexpr int PIPELINE_STAGES = 3;
    struct parsed_instruction {
        int row, col, iters;
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
    }
    __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id];
    }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[6];
    }
    __device__ static inline semaphore &outputs_shared(state<config> &s) {
        return s.semaphores()[7];
    }
    __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset);
    }
    __device__ static inline int get_b_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset + 2);
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
            }
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_finished(s, i), 4); // Inputs finished.
            }
            init_semaphore(outputs_arrived(s), 1); // outputs arrived.
            init_semaphore(outputs_shared(s), 4); // outputs shared.
            return 8;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int a_page = get_a_page(s, pipeline_stage, j);
                    if(i < PIPELINE_STAGES) s.wait_page_ready(a_page);
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    warp::tma::load_async(a, g.A, {inst.row+j, i}, inputs_arrived(s, pipeline_stage));
                }
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int b_page = get_b_page(s, pipeline_stage, j);
                    if(i < PIPELINE_STAGES) s.wait_page_ready(b_page);
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    warp::tma::load_async(b, g.B, {inst.col+j, i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                wait(inputs_finished(s, i), get_phasebit<1>(semaphore_bitfield, i));
            }
            arrive(outputs_arrived(s), 1);
        }
    };
    struct launcher { // launches mma's
        // Uses one minipage, and 4*iters full pages.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            s.wait_tensor_ready();
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for(int i = 0; i < inst.iters; i++) {
                int pipeline_stage = i%PIPELINE_STAGES;
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                update_phasebit<0>(semaphore_bitfield, pipeline_stage);
                if(laneid() < 4) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                    st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                    if(i == 0) mm <transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                    else       mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }
            warp::sync();
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            int store_page = warpgroup::groupid();
            wait(outputs_arrived(s), 0);
            if(warpid() >= 4 && warpid() < config::NUM_PAGES) warp::arrive(s.page_finished[s.pid(warpid())], config::NUM_CONSUMER_WARPS);
            st_fp8e4m3<128, 128> &store_buffer = s.pages[store_page].template as_st<fp8e4m3>();
            // Great, now we can start the store.
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(store_page*128);
            rt_fl<32, 128> acc_rt;
            rt_fp8e4m3<32, 128> acc_fp8;
            warpgroup::load_async(acc_rt, accumulator);
            warp::copy(acc_fp8, acc_rt);
            tensor_load_wait();
            warp::arrive(s.tensor_finished);
            warpgroup::store(store_buffer, acc_fp8);
            warpgroup::sync(warpgroup::groupid());
            warpgroup::arrive(outputs_shared(s));
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            wait(outputs_shared(s), 0);
            int r = laneid()/2, c = laneid()%2;
            if(laneid() < 4) {
                int page = s.pid(2*r+c);
                st_fp8e4m3<128, 128> &output = s.pages[page].template as_st<fp8e4m3>();
                tma::store_async(g.C, output, {inst.row+r, inst.col+c});
                tma::store_async_read_wait();
                warp::arrive(s.page_finished[page], config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kernel<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}