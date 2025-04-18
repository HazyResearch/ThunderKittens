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
        return s.semaphores()[id+PIPELINE_STAGES*2+4];
    }
    __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset);
    }
    __device__ static inline int get_b_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset + 2);
    }
    __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
        return s.pid(((inst.iters+2)%PIPELINE_STAGES)*4 + offset);
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            parsed_instruction inst{instruction};
            if(query == 0) return 12;
            else return ((query-1)+(inst.iters%PIPELINE_STAGES)*4)%12;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
                init_semaphore(inputs_finished(s, i), 4); // Inputs finished.
            }
            for(int i = 0; i < 4; i++) {
                init_semaphore(outputs_arrived(s, i), 1); // outputs arrived.
                init_semaphore(outputs_shared(s, i), 1); // outputs shared.
            }
            s.record(1);
            return 2*PIPELINE_STAGES + 8;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            warp::arrive(s.page_finished[s.pid(12)], config::NUM_CONSUMER_WARPS); // Release the unused page immediately.
            // if(laneid() == 0) printf("Thread %d, instruction %d, releasing logical page %d, physical page %d\n", threadIdx.x, s.instruction_index, 12, s.pid(12));
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                if(laneid() == 0 && i < 48) s.record(16+i);
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
                if(laneid() < 2) {
                    int a_page = get_a_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    tma::load_async(a, g.A, {inst.row+laneid(), i}, inputs_arrived(s, pipeline_stage));
                }
                if(laneid() < 2) {
                    int b_page = get_b_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                    }
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    tma::load_async(b, g.B, {inst.col+laneid(), i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();
            if(laneid() >= 28) {
                for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    int release_lid = pipeline_stage*4 + laneid() - 28;
                    int release_pid = s.pid(release_lid);
                    arrive(s.page_finished[release_pid], config::NUM_CONSUMER_WARPS);
                    // printf("Thread %d, instruction %d, releasing logical page %d, physical page %d\n", threadIdx.x, s.instruction_index, release_lid, release_pid);
                }
            }
        }
    };
    struct launcher { // launches mma's
        // Uses one minipage, and 4*iters full pages.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            // if(inst.iters == 1) asm volatile("trap;");
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            s.wait_tensor_ready();
            if(laneid() == 0) s.record(72);
            if(laneid() < 4) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
            }
            update_phasebit<0>(semaphore_bitfield, pipeline_stage);
            pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
            for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                if(laneid() == 0 && i < 48) s.record(72+i);
                if(laneid() < 4) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                    st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                    mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }
            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            if(laneid() == 0 && inst.iters <= 48) s.record(72+inst.iters-1);
            if(laneid() < 4) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
            }
            warp::sync();
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int groupid = warpgroup::groupid();
            wait(outputs_arrived(s, groupid), 0);
            if(group<16>::laneid() == 0) s.record(125);
            st_fp8e4m3<128, 128> &store_buffer = s.pages[get_store_page(s, inst, groupid)].template as_st<fp8e4m3>();
            // Great, now we can start the store.
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(groupid*128);
            rt_fl<32, 128> acc_rt;
            rt_fp8e4m3<32, 128> acc_fp8;
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
            if(laneid() < 4) {
                wait(outputs_shared(s, laneid()), 0);
                if(laneid() == 0) s.record(126);
                int store_page = get_store_page(s, inst, laneid());
                st_fp8e4m3<128, 128> &output = s.pages[store_page].template as_st<fp8e4m3>();
                tma::store_async(g.C, output, {inst.row+laneid()/2, inst.col+laneid()%2});
                tma::store_async_read_wait();
                arrive(s.page_finished[store_page], config::NUM_CONSUMER_WARPS);
                // printf("Thread %d, instruction %d, releasing logical page %d, physical page %d\n", threadIdx.x, s.instruction_index, ((inst.iters+2)%PIPELINE_STAGES)*4 + laneid(), store_page);
            }
            warp::sync();
            if(laneid() == 0) s.record(127);
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