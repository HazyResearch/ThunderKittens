#include <iostream>

#include "kittens.cuh"
#include "vm/vm.cuh" // Use the Kittens Virtual Machine framework
#include <limits>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

// --- Constants ---
constexpr int GQA_REDUCTION_OPCODE = 2;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;

using l_partial_sv = sv_fl<16>; //  (only index [0] is relevant)

using o_sv = sv_fl<HEAD_DIM>;
using o_rv = rv_fl<HEAD_DIM>;
using o_vector_rt = rt_fl<16, HEAD_DIM>;
using o_partial_st = st_fl<16, HEAD_DIM>; // Store O partials (only row [0] is relevant)
using o_final_st = st_bf<16, HEAD_DIM>; // Store final O output (only row [0] is relevant)
using o_final_sv = sv_bf<HEAD_DIM>;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    // Input Partial LSE:
    using l_partial_layout = gl<float, 1, 1, NUM_Q_HEADS, -1, l_partial_sv>;
    // Input Partial O:
    using o_partial_layout = gl<float, 1, NUM_Q_HEADS, -1, HEAD_DIM, o_sv>;
    // Final Output O:
    using o_final_layout = gl<bf16, 1, NUM_Q_HEADS, 1, HEAD_DIM, o_final_sv>;

    instruction_layout instructions;
    timing_layout timings;
    l_partial_layout L_partials;     // Input: Global partial LSE tensor
    o_partial_layout O_partials;     // Input: Global partial O tensor
    o_final_layout O_final;          // Output: Global final O tensor

    dim3 grid() { return dim3(NUM_Q_HEADS); } // One block per Q head
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_gqa_reduction_op {
    static constexpr int opcode = GQA_REDUCTION_OPCODE;
    static constexpr int NUM_STAGES = 4;

    // --- Instruction Parsing  ---
    // [0] = opcode (2)
    // [1] = num_partials
    struct parsed_instruction {
        int num_partials;
        int q_head_idx; // Implicitly blockIdx.x
        __device__ inline parsed_instruction(state<config> &s) {
            num_partials = s.instruction()[1];
            q_head_idx = blockIdx.x;
        }
    };

    // --- Semaphore Access Helpers ---
    __device__ static inline semaphore &L_partial_arrived(state<config> &s, int stage) { return s.semaphores()[stage * 2]; }
    __device__ static inline semaphore &O_partial_arrived(state<config> &s, int stage) { return s.semaphores()[stage * 2 + 1]; }
    __device__ static inline semaphore &L_partial_finished(state<config> &s, int stage) { return s.semaphores()[NUM_STAGES * 2 + stage * 2]; }
    __device__ static inline semaphore &O_partial_finished(state<config> &s, int stage) { return s.semaphores()[NUM_STAGES * 2 + stage * 2 + 1]; }
    __device__ static inline semaphore &final_O_ready(state<config> &s) { return s.semaphores()[NUM_STAGES * 4]; }

    // --- Shared Memory Page Management Helpers ---
    static constexpr int PARTIALS_PAGE = 0; // Page for pipelined partial O/L
    static constexpr int FINAL_O_PAGE = 1;  // Page for final O storage
    __device__ static inline void wait_partials_page(state<config> &s) { s.wait_page_ready(s.pid(PARTIALS_PAGE)); }
    __device__ static inline void wait_final_o_page(state<config> &s) { s.wait_page_ready(s.pid(FINAL_O_PAGE)); }
    __device__ static inline void finish_partials_page(state<config> &s) {
        if (warp::laneid() == 0) arrive(s.page_finished[s.pid(PARTIALS_PAGE)], config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_final_o_page(state<config> &s) {
        if (warp::laneid() == 0) arrive(s.page_finished[s.pid(FINAL_O_PAGE)], config::NUM_CONSUMER_WARPS);
    }

    // --- Shared Memory Access Helpers ---
    __device__ static inline l_partial_sv &get_L_partial_smem(state<config> &s, int stage) {
        int pid = s.pid(PARTIALS_PAGE);
        char *base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        size_t offset = stage * (sizeof(l_partial_sv) + sizeof(o_sv));
        return *reinterpret_cast<l_partial_sv*>(base_ptr + offset);
    }
     __device__ static inline o_sv &get_O_partial_smem(state<config> &s, int stage) {
        int pid = s.pid(PARTIALS_PAGE);
        char *base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        size_t offset = stage * (sizeof(l_partial_sv) + sizeof(o_sv)) + sizeof(l_partial_sv);
        return *reinterpret_cast<o_sv*>(base_ptr + offset);
    }
    __device__ static inline o_final_sv &get_O_final_smem(state<config> &s) {
        int pid = s.pid(FINAL_O_PAGE);
        return *reinterpret_cast<o_final_sv*>(s.pages[pid].data);
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(L_partial_arrived(s, i), 0, 1);
                init_semaphore(O_partial_arrived(s, i), 0, 1);
                init_semaphore(L_partial_finished(s, i), 0, 1);
                init_semaphore(O_partial_finished(s, i), 0, 1);
            }
            init_semaphore(final_O_ready(s), 0, 1);
            return 4 * NUM_STAGES + 1;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            
            if (laneid >= 2 && laneid < config::NUM_PAGES) arrive(s.page_finished[s.pid(laneid)], config::NUM_CONSUMER_WARPS);
            if (laneid == 0) {
                wait_partials_page(s);
                wait_final_o_page(s);

                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    l_partial_sv &L_smem = get_L_partial_smem(s, stage);
                    o_sv &O_smem = get_O_partial_smem(s, stage);

                    if (i >= NUM_STAGES) {
                        wait(L_partial_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(O_partial_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    // Load L_partial[q_head_idx, i]
                    L_smem.data[0] = g.L_partials.raw_ptr[(inst.q_head_idx * g.L_partials.cols()) + i];
                    for (int i = 1; i < 16; ++i) {
                        L_smem.data[i] = 0;
                    }
                    // tma::expect(L_partial_arrived(s, stage), L_smem);
                    // tma::load_async<cache_policy::EVICT_FIRST>(L_smem, g.L_partials, {0, 0, inst.q_head_idx, i}, L_partial_arrived(s, stage));

                    // Load O_partial[q_head_idx, i] into row 0 of the SMEM tile
                    tma::expect(O_partial_arrived(s, stage), O_smem);
                    tma::load_async<cache_policy::EVICT_FIRST>(O_smem, g.O_partials, {0, inst.q_head_idx, i, 0}, O_partial_arrived(s, stage));
                }
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if (warpid() == 0) {
                o_rv O_final_reg;        // Accumulator O vector
                o_rv O_partial_reg;      // Loaded partial O vector

                float L_final_reg;          // Accumulator LSE (log-sum-exp)
                float L_partial_reg;        // Loaded partial LSE
                float L_max_reg;            // Max in the current step
                float L_max_accum_reg;      // Overall max LSE accumulator
                float scale_final_reg;
                float scale_partial_reg;

                warp::zero(O_final_reg);
                L_final_reg = -INFINITY;
                L_max_accum_reg = -INFINITY;

                // --- Reduction Pipeline ---
                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    warp::wait(O_partial_arrived(s, stage), (i / NUM_STAGES) % 2);

                    l_partial_sv &L_smem = get_L_partial_smem(s, stage);
                    o_sv &O_smem = get_O_partial_smem(s, stage);

                    // Load L_partial_reg
                    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&L_smem.data[0]));
                    move<float>::lds(L_partial_reg, src_ptr);
                    // Load O_partial_reg
                    warp::load(O_partial_reg, O_smem);
                    
                    // Update L_max_reg accumulator
                    L_max_reg = max(L_final_reg, L_partial_reg);

                    // Calculate scales based on L_max_reg
                    scale_final_reg = exp2f(L_final_reg - L_max_reg);
                    scale_partial_reg = exp2f(L_partial_reg - L_max_reg);

                    // Update O accumulator
                    warp::mul(O_final_reg, O_final_reg, scale_final_reg);
                    warp::mul(O_partial_reg, O_partial_reg, scale_partial_reg);
                    warp::add(O_final_reg, O_final_reg, O_partial_reg);

                    // Update LSE accumulator
                    float sum_scales = scale_final_reg + scale_partial_reg;
                    L_final_reg = L_max_reg + log2f(sum_scales);

                    warp::arrive(L_partial_finished(s, stage));
                    warp::arrive(O_partial_finished(s, stage));
                }
                finish_partials_page(s);

                warp::div(O_final_reg, O_final_reg, exp2f(L_final_reg - L_max_reg));

                o_final_sv &O_final_smem = get_O_final_smem(s);
                warp::store(O_final_smem, O_final_reg);
                warp::sync();

                warp::arrive(final_O_ready(s));
            }
        }
    };

    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            
            if (warp::laneid() == 0) {
                o_final_sv &O_final_smem = get_O_final_smem(s);
                wait(final_O_ready(s), 0);
    
                tma::store_async<cache_policy::NORMAL>(g.O_final, O_final_smem, {0, inst.q_head_idx, 0, 0});
                tma::store_async_read_wait();
                finish_final_o_page(s);
            }
            warp::sync();
         }
    };
};

#include "pyutils/pyutils.cuh"

// --- Python Bindings ---
PYBIND11_MODULE(gqa_reduction, m) {
    m.doc() = "GQA Reduction VM Operation (Compliant Types)";
    kittens::py::bind_kernel<kvm<config, globals, rope_gqa_reduction_op<config>>>(m, "gqa_reduction",
        &globals::instructions,
        &globals::timings,
        &globals::L_partials,
        &globals::O_partials,
        &globals::O_final
    );
}