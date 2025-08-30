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
[2] = Col offset of C, in units of 256
[3] = K reduction dimension, in units of 128
*/

using a_tile = st_fp8e4m3<128, 128>;
using b_tile = st_fp8e4m3<256, 128>;
using c_tile = st_fp8e4m3<128, 256>;

static constexpr int SM_COUNT = 148;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, a_tile, b_tile, c_tile>;
    instruction_layout instructions;
    timing_layout timings;
    fp8_matrix A, B, C;
    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 1;
    static constexpr int PIPELINE_STAGES = 3;
    struct parsed_instruction {
        int row, col, iters;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            row = instruction[1];
            col = instruction[2];
            iters = instruction[3];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
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
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2];
    }
    __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2+2];
    }
    __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
        return stage*4 + offset;
    }
    __device__ static inline int get_b_page(state<config> &s, int stage) {
        return stage*4 + 2; // single mega page for b (32KB)
    }
    __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
        return ((inst.iters+2)%PIPELINE_STAGES)*4 + offset*2; // 32KB megapages
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
                init_semaphore(inputs_finished(s, i), 2); // Inputs finished.
            }
            for(int i = 0; i < 2; i++) {
                init_semaphore(outputs_arrived(s, i), 1); // outputs arrived.
                init_semaphore(outputs_shared(s, i), 1); // outputs shared.
            }
            return 2*PIPELINE_STAGES + 4;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            s.warp_finish_page(12, config::NUM_CONSUMER_WARPS); // Release the unused page immediately.
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                if(laneid() < 2) {
                    int a_page = get_a_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    a_tile &a = s.pages[a_page].template as_st<fp8e4m3>();
                    tma::load_async(a, g.A, {inst.row+laneid(), i}, inputs_arrived(s, pipeline_stage));
                } else if (laneid() == 2) {
                    int b_page = get_b_page(s, pipeline_stage);
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                        s.wait_page_ready(b_page+1); // because b_page is a megapage
                    }
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                    tma::load_async(b, g.B, {inst.col/2, i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();
            if(laneid() >= 28) {
                for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    int release_pid = pipeline_stage*4 + laneid() - 28;
                    s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };
    struct launcher { // launches mma's
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;

            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            s.wait_tensor_ready();
            if(laneid() < 2) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 256>>(laneid()*256);
                a_tile &a = s.pages[get_a_page(s, pipeline_stage, laneid())].template as_st<fp8e4m3>();
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
            }
            update_phasebit<0>(semaphore_bitfield, pipeline_stage);
            pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
            
            for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                if(laneid() < 2) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 256>>(laneid()*256);
                    a_tile &a = s.pages[get_a_page(s, pipeline_stage, laneid())].template as_st<fp8e4m3>();
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                    mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }

            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            if(laneid() < 2) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 256>>(laneid()*256);
                a_tile &a = s.pages[get_a_page(s, pipeline_stage, laneid())].template as_st<fp8e4m3>();
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
            }
            warp::sync();
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int groupid = warpgroup::groupid(); // 2 warpgroups
            wait(outputs_arrived(s, groupid), 0);
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 256>>(groupid*256);
            c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[get_store_page(s, inst, groupid)].data);
            rt_fl<32, 256> acc_rt;
            rt_fp8e4m3<32, 256> acc_fp8;
            warpgroup::load_async(acc_rt, accumulator);
            warp::copy(acc_fp8, acc_rt);
            tensor_load_wait();
            warp::arrive(s.tensor_finished);
            warpgroup::store(store_buffer, acc_fp8);
            warpgroup::sync(groupid);
            warpgroup::arrive(outputs_shared(s, groupid));
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if(laneid() < 2) {
                wait(outputs_shared(s, laneid()), 0);
                int store_page = get_store_page(s, inst, laneid());
                c_tile &output = *reinterpret_cast<c_tile *>(s.pages[get_store_page(s, inst, laneid())].data);
                tma::store_async(g.C, output, {inst.row+laneid(), inst.col/2});
                tma::store_async_read_wait();
                s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
                s.finish_page(store_page + 1, config::NUM_CONSUMER_WARPS); // because store_page is megapage
            }
            warp::sync();
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kvm<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}