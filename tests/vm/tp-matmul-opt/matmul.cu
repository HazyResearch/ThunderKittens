#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int SM_COUNT = 148;

using config = default_config;
struct globals {
    constexpr static int num_devices = 8;
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using barrier_layout = gl<uint, 1, 1, -1, -1>; // num_rows x iters
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, st_fp8e4m3<128, 128>>;
    instruction_layout instructions;
    gl_array<barrier_layout, num_devices> barriers;
    timing_layout timings;
    gl_array<fp8_matrix, num_devices> A0s; // A is shared across devices.
    gl_array<fp8_matrix, num_devices> A1s;
    fp8_matrix B;
    fp8_matrix C;
    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 725;
    static constexpr int PIPELINE_STAGES = 3;

    struct parsed_instruction {
        int row, local_row, local_col, iters, dev_idx, next_dev_idx, phasebit;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            row = instruction[1];
            local_row = instruction[2];
            local_col = instruction[3];
            iters = instruction[4];
            dev_idx = instruction[5];
            next_dev_idx = instruction[6];
            phasebit = instruction[7]; // when 0, use A0 for compute; when 1, use A1 for compute
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

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
                init_semaphore(inputs_finished(s, i), 6); // Inputs finished -- 4 by consumer + 2 by storer
            }
            for(int i = 0; i < 4; i++) {
                init_semaphore(outputs_arrived(s, i), 1); // outputs arrived.
                init_semaphore(outputs_shared(s, i), 1); // outputs shared.
            }
            return 2*PIPELINE_STAGES + 8;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            if (laneid() == 0) s.record(0);
            warp::arrive(s.page_finished[s.pid(12)], config::NUM_CONSUMER_WARPS); // Release the unused page immediately.

            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                if (laneid() == 0) s.record(10 + i);
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
                if(laneid() < 2) {
                    while (*(volatile int *)&g.barriers[inst.dev_idx][{inst.row + laneid(), i}] < 1)
                        __nanosleep(20);
                    int a_page = get_a_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    if (inst.phasebit == 0)
                        tma::load_async(a, g.A0s[inst.dev_idx], {inst.local_row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                    else
                        tma::load_async(a, g.A1s[inst.dev_idx], {inst.local_row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                }
                if(laneid() < 2) {
                    int b_page = get_b_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                    }
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    tma::load_async(b, g.B, {inst.local_col + laneid(), i}, inputs_arrived(s, pipeline_stage));
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
                }
            }
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            if (laneid() == 0) s.record(50);
            s.wait_tensor_ready();

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
                if (laneid() == 0) s.record(50 + i);
                if(laneid() < 4) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                    st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                    mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }

            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            if (laneid() == 0) s.record(50 + inst.iters - 1);
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

            st_fp8e4m3<128, 128> &store_buffer = s.pages[get_store_page(s, inst, groupid)].template as_st<fp8e4m3>();
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

                int store_page = get_store_page(s, inst, laneid());
                st_fp8e4m3<128, 128> &output = s.pages[store_page].template as_st<fp8e4m3>();
                tma::store_async(g.C, output, {inst.row+laneid()/2, inst.local_col+laneid()%2});
                tma::store_async_read_wait();
                arrive(s.page_finished[store_page], config::NUM_CONSUMER_WARPS);
            } else if (4 <= laneid() && laneid() < 6) { // redundant, but prevents stupid mistakes in the future
                uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
                int pipeline_stage = 0;
                int relative_laneid = laneid() - 4;
                for(int i = 0; i < inst.iters; ++i, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                    st_fp8e4m3<128, 128> &a_page = s.pages[get_a_page(s, pipeline_stage, relative_laneid)].template as_st<fp8e4m3>();
                    if (inst.phasebit == 0)
                        tma::store_async(g.A1s[inst.next_dev_idx], a_page, {inst.local_row + relative_laneid, i});
                    else
                        tma::store_async(g.A0s[inst.next_dev_idx], a_page, {inst.local_row + relative_laneid, i});
                    // tma::store_async_read_wait();
                    tma::store_async_wait();
                    arrive(inputs_finished(s, pipeline_stage));
                    atomicAdd(&g.barriers[inst.next_dev_idx][{inst.row + laneid(), i}], 1);
                }
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
        &globals::barriers,
        &globals::timings,
        &globals::A0s,
        &globals::A1s,
        &globals::B,
        &globals::C
    );
}
