#include <iostream>

#include "kittens.cuh"
#include "vm/vm.cuh"

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

constexpr int NUM_BLOCKS = 148;
constexpr int ROPE_GQA_PARTIAL_OPCODE = 2;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using bf16_matrix = gl<bf16, 1, -1, -1, -1, st_bf<128, 128>>; // assume single batch
    instruction_layout instructions;
    timing_layout timings;
    bf16_matrix Q, K_c, V_c, O;
    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_gqa_partial_op {
    static constexpr int opcode = ROPE_GQA_PARTIAL_OPCODE;
    static constexpr int PIPELINE_STAGES = 3;

    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
    }

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id];
    }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2];
    }
    __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2+4];
    }
    __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset);
    }
    __device__ static inline int get_b_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset + 2);
    }

    __device__ static inline void debug(const char* msg) {
        printf("Warp ID %d, Lane ID %d: %s\n", warpid(), laneid(), msg);
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            // for(int i = 0; i < PIPELINE_STAGES; i++) {
            //     init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
            //     init_semaphore(inputs_finished(s, i), 4); // Inputs finished.
            // }
            // for(int i = 0; i < 4; i++) {
            //     init_semaphore(outputs_arrived(s, i), 1); // outputs arrived.
            //     init_semaphore(outputs_shared(s, i), 1); // outputs shared.
            // }
            return 2*PIPELINE_STAGES + 8;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            // debug("Loader run");
            // parsed_instruction inst = parse_instruction(g, s);

            // uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

            // int pipeline_stage = 0;
            // for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
            //     wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
            //     warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int Q_page = get_a_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(Q_page);
            //         }
            //         st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(a, g.A, {inst.row+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int b_page = get_b_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(b_page);
            //         }
            //         st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(b, g.B, {inst.col+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            // }
            // warp::sync();
        }
    };
    struct launcher { // launches mma's
        static __device__ void run(const globals &g, state<config> &s) {
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(rope_gqa_partial, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel<config, globals, rope_gqa_partial_op<config>>>(m, "rope_gqa_partial",
        &globals::instructions,
        &globals::timings,
        &globals::Q,
        &globals::K_c,
        &globals::V_c,
        &globals::O
    );
}
