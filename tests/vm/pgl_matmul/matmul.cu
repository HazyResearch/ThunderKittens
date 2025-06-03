#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int M_BLOCK = 128;
constexpr int N_BLOCK = 256;
constexpr int K_BLOCK = 128;

using a_tile = st_fp8e4m3<M_BLOCK / 2, K_BLOCK>; // 2 consumer warpgroups
using b_tile = st_fp8e4m3<N_BLOCK, K_BLOCK>;
using c_tile = st_fp8e4m3<M_BLOCK / 2, N_BLOCK>;

static constexpr int SM_COUNT = 132;

struct config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 1024;
    static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 160;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

struct globals {
    static constexpr int num_devices = 4;

    using instruction_layout = pgl<gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>, num_devices, false>;
    using timing_layout = pgl<gl<int, 1, -1, -1, config::TIMING_WIDTH>, num_devices, false>;
    using fp8_matrix = pgl<gl<fp8e4m3, 1, -1, -1, -1, a_tile, b_tile, c_tile>, num_devices, true, true>;

    instruction_layout instructions;
    timing_layout timings;
    fp8_matrix A, B, C;

    int dev_idx;
    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 1;
    static constexpr int PIPELINE_STAGES = 4;

    static_assert(config::NUM_PAGES == 13);
    static_assert(config::PAGE_SIZE == 16384);
    static_assert(PIPELINE_STAGES >= 2);
    static_assert(config::NUM_CONSUMER_WARPS == 8);
    
    struct parsed_instruction {
        int row, col, num_iters;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            /*
                Instruction format:
                [0] = opcode
                [1] = Row offset of C, in units of M_BLOCK
                [2] = Col offset of C, in units of N_BLOCK
                [3] = K reduction dimension, in units of K_BLOCK
            */
            row = instruction[1];
            col = instruction[2];
            num_iters = instruction[3];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) { return s.semaphores()[stage+PIPELINE_STAGES]; }
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int consumer_id) { return s.semaphores()[consumer_id+PIPELINE_STAGES*2]; }
    __device__ static inline int get_a_page(state<config> &s, int stage) { return stage*3; }
    __device__ static inline int get_b_page(state<config> &s, int stage) { return stage*3 + 1; }
    __device__ static inline int get_store_page(state<config> &s, int consumer_id) {
        return (PIPELINE_STAGES-1)*3 + consumer_id; // use 2 pages from the last stage
    }

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
                init_semaphore(outputs_arrived(s, i), 4);
            }
            return 2*PIPELINE_STAGES + 2;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();

            if (laneid >= PIPELINE_STAGES*3 && laneid < config::NUM_PAGES) {
                s.wait_page_ready(laneid);
                s.finish_page(laneid, config::NUM_CONSUMER_WARPS); // release unused pages immediately
            }

            int pipeline_stage = 0;
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for (int i = 0; i < inst.num_iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                if (laneid < 2) {
                    int a_page = get_a_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * laneid);
                    tma::load_async(a, g.A[g.dev_idx], {inst.row*2 + laneid, i}, inputs_arrived(s, pipeline_stage));
                } else if (laneid == 2) {
                    int b_page = get_b_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                        s.wait_page_ready(b_page + 1); // because b_page is a megapage
                    }
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                    tma::load_async(b, g.B[g.dev_idx], {inst.col, i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();

            for (int i = 0; i < PIPELINE_STAGES; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                if ((laneid < 3 && pipeline_stage < PIPELINE_STAGES - 1) || (laneid == 2 && pipeline_stage == PIPELINE_STAGES - 1)) {
                    int release_pid = pipeline_stage*3 + laneid;
                    s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };
    struct launcher { // no warpgroup-level tensor cores in H100s
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warpgroup::laneid();
            int warpid = warpgroup::warpid();
            int groupid = warpgroup::groupid();

            rt_fl<M_BLOCK / config::NUM_CONSUMER_WARPS, N_BLOCK> acc_fl;
            warp::zero(acc_fl);

            int pipeline_stage = 0;
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for (int i = 0; i < inst.num_iters; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                int a_page = get_a_page(s, pipeline_stage);
                int b_page = get_b_page(s, pipeline_stage);
                a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * groupid);
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                warpgroup::mma_ABt(acc_fl, a, b);
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, pipeline_stage));
            }

            rt_fp8e4m3<M_BLOCK / config::NUM_CONSUMER_WARPS, N_BLOCK> acc_fp8;
            warp::copy(acc_fp8, acc_fl);

            int store_page = get_store_page(s, groupid);
            c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
            warpgroup::store(store_buffer, acc_fp8);
            __syncwarp();
            warp::arrive(outputs_arrived(s, groupid));
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();

            if (laneid < 2) {
                wait(outputs_arrived(s, laneid), 0);
                int store_page = get_store_page(s, laneid);
                c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
                tma::store_async(g.C[g.dev_idx], store_buffer, {inst.row*2 + laneid, inst.col});
                tma::store_async_read_wait();
                s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
            }
            warp::sync();
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_multigpu_kernel<kvm<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}