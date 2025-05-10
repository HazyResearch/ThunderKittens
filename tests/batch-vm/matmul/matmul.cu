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
Instruction format: (ASSUMING HOST UPDATES UNITS)
[0] = opcode
[1] = Row offset of C, in units of 64
[2] = Col offset of C, in units of 128
[3] = K reduction dimension, in units of 128

Outputs a 128 x 128 matrix of C 
*/

using a_tile = st_bf<64, 128>;    // 16KB
using b_tile = st_bf<128, 128>;   // 32KB
using c_tile = st_bf<64, 128>;    // 16KB

/*
Could also try... 
using a_tile = st_bf<128, 64>;   // 16KB
using b_tile = st_bf<256, 64>;   // 32KB
using c_tile = st_bf<128, 256>;  // 64KB
*/

static constexpr int SM_COUNT = 148;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using bf16_matrix = gl<bf16, 1, 1, -1, -1, a_tile, b_tile, c_tile>;
    instruction_layout instructions;
    timing_layout timings;
    bf16_matrix A, B, C;
    
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
            row = instruction[1]; // units of 64 rows
            col = instruction[2]; // units of 128 cols
            iters = instruction[3]; // units of 128 K-dim
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
        return stage*4 + 2;
    }
    __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
        return ((inst.iters+2)%PIPELINE_STAGES)*4 + offset*2;
    }
    // --- END PAGE GETTERS ---

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);
                init_semaphore(inputs_finished(s, i), 2);
            }
            for(int i = 0; i < 2; i++) {
                init_semaphore(outputs_arrived(s, i), 1);
                init_semaphore(outputs_shared(s, i), 1);
            }
            return 2*PIPELINE_STAGES + (2 * 2); // Total semaphores initialized
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(BLUE_TEXT "Loader Passed stage %d\n" RESET_TEXT, pipeline_stage);
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                if(laneid() < 2) {
                    int a_page = get_a_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    a_tile &a = *reinterpret_cast<a_tile *>(s.pages[a_page].data);
                    tma::load_async(a, g.A, {inst.row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                } else if (laneid() == 2) {
                    int b_page = get_b_page(s, pipeline_stage);
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                        s.wait_page_ready(b_page+1);
                    }
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                    tma::load_async(b, g.B, {inst.col, i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync(); // Ensure all loads are issued
            // if (laneid() == 0) printf(BLUE_TEXT "Loader finished issuing loads\n" RESET_TEXT);

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
            uint32_t semaphore_bitfield = 0xFFFF0000;
            int pipeline_stage = 0;

            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
            s.wait_tensor_ready();
            if(laneid() < 2) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
            }
            // if (laneid() == 0) printf(GREEN_TEXT "Finished first mma\n" RESET_TEXT);
            update_phasebit<0>(semaphore_bitfield, pipeline_stage);
            pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
            
            for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
                if(laneid() < 2) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                    a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                    mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }
            
            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);

            if(laneid() < 2) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
            }
            warp::sync();
            // if (laneid() == 0) printf(RED_TEXT "Finished launcher\n" RESET_TEXT);

        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int groupid = warpgroup::groupid();
            // if (groupid < 2 && warpgroup::warpid() < 2)
            if (groupid < 2)
            {
                wait(outputs_arrived(s, groupid), 0);
    
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(groupid, 0);
                // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, groupid * 128);
                rt_fl<16, 128> acc_rt;
                rt_bf<16, 128> acc_bf16;
                
                warpgroup::load_async(acc_rt, accumulator);
                tensor_load_wait();
                warp::copy(acc_bf16, acc_rt);
                warp::arrive(s.tensor_finished);
                
                int store_page_id = get_store_page(s, inst, groupid);
                c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page_id].data);
                warpgroup::store(store_buffer, acc_bf16);
                warpgroup::sync(groupid);
                warpgroup::arrive(outputs_shared(s, groupid));
            }
        }
    };
    /*
    rt_fl<32, 128> acc_rt;
    rt_bf<32, 128> acc_bf16;

    auto src_subtile = accumulator.template subtile<tt<float, 32, 128>>(32*warpgroup::warpid(), 0);
    warp::load_async(acc_rt, src_subtile);
    
    // warpgroup::load_async(acc_rt, accumulator);
    */
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if (laneid() < 2) {
                wait(outputs_shared(s, laneid()), 0);
                int store_page = get_store_page(s, inst, laneid());
                c_tile &output = *reinterpret_cast<c_tile *>(s.pages[get_store_page(s, inst, laneid())].data);
                tma::store_async(g.C, output, {inst.row+laneid(), inst.col});
                tma::store_async_read_wait();
                s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
                s.finish_page(store_page+1, config::NUM_CONSUMER_WARPS); // not used but should still be released 
            }
            warp::sync();
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "bf16 matmul python module with KVM";
    kittens::py::bind_kernel<kvm<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}