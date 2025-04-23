#include <iostream>
#include "kittens.cuh"
#include "vm/vm.cuh"
#include <limits>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int Q_HEADS_PER_INSTRUCTION = 4;
constexpr int GQA_PARTIAL_OPCODE = 1;
constexpr int GQA_REDUCTION_OPCODE = 2;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;

static_assert(NUM_Q_HEADS % Q_HEADS_PER_INSTRUCTION == 0, 
    "NUM_Q_HEADS must be divisible by Q_HEADS_PER_INSTRUCTION");

using l_partial_sv = sv_fl<16>;
using o_sv = sv_fl<HEAD_DIM>;
using o_rv = rv_fl<HEAD_DIM>;
using o_final_sv = sv_bf<HEAD_DIM>;

using config = default_config;

struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using bar_layout = gl<uint, 1, -1, 6, NUM_Q_HEADS + 2 * NUM_KV_HEADS>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    // Input Partial LSE:
    using l_partial_layout = gl<float, 1, 1, NUM_Q_HEADS, -1, l_partial_sv>;
    // Input Partial O:
    using o_partial_layout = gl<float, 1, NUM_Q_HEADS, -1, HEAD_DIM, o_sv>;
    // Final Output O:
    using o_final_layout = gl<bf16, 1, NUM_Q_HEADS, 1, HEAD_DIM, o_final_sv>;

    instruction_layout instructions;
    bar_layout barriers; 
    timing_layout timings;
    l_partial_layout L_partials;     // Input: Global partial LSE tensor
    o_partial_layout O_partials;     // Input: Global partial O tensor
    o_final_layout O_final;          // Output: Global final O tensor

    // One block for every 4 Q heads
    dim3 grid() { return dim3(NUM_Q_HEADS / Q_HEADS_PER_INSTRUCTION); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_gqa_reduction_op {
    static constexpr int opcode = GQA_REDUCTION_OPCODE;
    static constexpr int NUM_STAGES = 2;

    // --- Instruction Parsing ---
    // Instruction format:
    // [0] = opcode (GQA_REDUCTION_OPCODE)
    // [1] = layer_idx
    // [2] = num_partials
    struct parsed_instruction {
        int layer_idx;
        int num_partials;
        int q_head_start_idx;
        __device__ inline parsed_instruction(state<config> &s) {
            layer_idx = s.instruction()[1];
            num_partials = s.instruction()[2];
            q_head_start_idx = blockIdx.x * 4;
        }
    };

    // --- Semaphore Access Helpers ---
    static constexpr int NUM_O_SEMAPHORES = Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2;
    static constexpr int NUM_L_SEMAPHORES = Q_HEADS_PER_INSTRUCTION * 2; 
    static constexpr int NUM_FINAL_O_SEMAPHORES = Q_HEADS_PER_INSTRUCTION;
    static constexpr int TOTAL_SEMAPHORES_NEEDED = NUM_O_SEMAPHORES + NUM_L_SEMAPHORES + NUM_FINAL_O_SEMAPHORES;
    static_assert(config::DYNAMIC_SEMAPHORES >= TOTAL_SEMAPHORES_NEEDED,
                  "Insufficient dynamic semaphores for specified config");

     __device__ static constexpr int O_partial_sem_idx(int q_head_local_idx, int stage, bool is_finished) {
        assert(q_head_local_idx >= 0 && q_head_local_idx < Q_HEADS_PER_INSTRUCTION);
        assert(stage >= 0 && stage < NUM_STAGES);
        return q_head_local_idx * (NUM_STAGES * 2) + stage * 2 + (is_finished ? 1 : 0);
    }
    __device__ static constexpr int L_partial_sem_idx(int q_head_local_idx, bool is_finished) {
         assert(q_head_local_idx >= 0 && q_head_local_idx < Q_HEADS_PER_INSTRUCTION);
         return NUM_O_SEMAPHORES + q_head_local_idx * 2 + (is_finished ? 1 : 0);
    }
    __device__ static constexpr int Final_O_ready_sem_idx(int q_head_local_idx) {
         assert(q_head_local_idx >= 0 && q_head_local_idx < Q_HEADS_PER_INSTRUCTION);
         return NUM_O_SEMAPHORES + NUM_L_SEMAPHORES + q_head_local_idx;
    }

    __device__ static inline semaphore &O_partial_arrived(state<config> &s, int q_head_local_idx, int stage) {
        return s.semaphores()[O_partial_sem_idx(q_head_local_idx, stage, false)];
    }
    __device__ static inline semaphore &O_partial_finished(state<config> &s, int q_head_local_idx, int stage) {
        return s.semaphores()[O_partial_sem_idx(q_head_local_idx, stage, true)];
    }
    __device__ static inline semaphore &L_partial_all_arrived(state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, false)];
    }
    __device__ static inline semaphore &L_partial_all_finished(state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, true)];
    }
    __device__ static inline semaphore &final_O_ready(state<config> &s, int q_head_local_idx) {
        return s.semaphores()[Final_O_ready_sem_idx(q_head_local_idx)];
    }

    // --- Shared Memory Page Management Helpers ---
    static constexpr int SHARED_DATA_PAGE = 0; // Use only the first logical page

    __device__ static inline void wait_shared_page(state<config> &s) {
        s.wait_page_ready(s.pid(SHARED_DATA_PAGE));
    }
    __device__ static inline void finish_shared_page(state<config> &s) {
        if (warp::laneid() == 0) {
             arrive(s.page_finished[s.pid(SHARED_DATA_PAGE)], config::NUM_CONSUMER_WARPS);
        }
    }

    // --- Shared Memory Layout and Access Helpers (Single Page) ---
    // Calculate the size needed for partials buffering
    static constexpr size_t size_per_head = sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv) + sizeof(o_final_sv);
    static constexpr size_t total_smem_needed = Q_HEADS_PER_INSTRUCTION * size_per_head;
    static_assert(total_smem_needed <= config::PAGE_SIZE,
        "Required shared memory exceeds configured page size.");

    __device__ static inline l_partial_sv &get_L_partial_smem(state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        return *reinterpret_cast<l_partial_sv*>(head_base_ptr);
    }
    
    __device__ static inline o_sv &get_O_partial_smem(state<config> &s, int q_head_local_idx, int stage) {
        assert(stage >= 0 && stage < NUM_STAGES);
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + stage * sizeof(o_sv);
        return *reinterpret_cast<o_sv*>(head_base_ptr + offset);
    }
    
    __device__ static inline o_final_sv &get_O_final_smem(state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv);
        return *reinterpret_cast<o_final_sv*>(head_base_ptr + offset);
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for (int q_head = 0; q_head < Q_HEADS_PER_INSTRUCTION; ++q_head) {
                for (int stage = 0; stage < NUM_STAGES; stage++) {
                    init_semaphore(O_partial_arrived(s, q_head, stage), 0, 1);
                    init_semaphore(O_partial_finished(s, q_head, stage), 0, 1);
                }
                init_semaphore(L_partial_all_arrived(s, q_head), 0, 1);
                init_semaphore(L_partial_all_finished(s, q_head), 0, 1);

                init_semaphore(final_O_ready(s, q_head), 0, 1);
            }
            return TOTAL_SEMAPHORES_NEEDED;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            if (laneid < Q_HEADS_PER_INSTRUCTION) {
                while (*(volatile int *)&g.barriers[{inst.layer_idx, GQA_PARTIAL_OPCODE, inst.q_head_start_idx}] != 1) {
                    __nanosleep(20);
                }

                wait_shared_page(s);

                l_partial_sv &L_smem = get_L_partial_smem(s, laneid);
                tma::expect(L_partial_all_arrived(s, laneid), L_smem);
                tma::load_async<cache_policy::EVICT_FIRST>(
                    L_smem, g.L_partials, {0, 0, inst.q_head_start_idx + laneid, 0}, L_partial_all_arrived(s, laneid));

                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    o_sv &O_smem = get_O_partial_smem(s, laneid, stage);

                    if (i >= NUM_STAGES) {
                        int prev_phase = (i / NUM_STAGES - 1) % 2;
                        wait(O_partial_finished(s, laneid, stage), prev_phase);
                    }

                    tma::expect(O_partial_arrived(s, laneid, stage), O_smem);
                    tma::load_async<cache_policy::EVICT_FIRST>(
                        O_smem, g.O_partials, {0, inst.q_head_start_idx + laneid, i, 0}, O_partial_arrived(s, laneid, stage));
                }
            } else if (laneid - (Q_HEADS_PER_INSTRUCTION - 1) < config::NUM_PAGES) {
                arrive(s.page_finished[s.pid(laneid - (Q_HEADS_PER_INSTRUCTION - 1))], config::NUM_CONSUMER_WARPS);
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

            if (warpid() < Q_HEADS_PER_INSTRUCTION) {
                int q_head_local_idx = warpid();

                o_rv accumulated_out;
                float accumulated_lse = -INFINITY;

                o_rv current_out;
                float current_lse;

                warp::zero(accumulated_out);

                warp::wait(L_partial_all_arrived(s, q_head_local_idx), 0);
                l_partial_sv &L_smem = get_L_partial_smem(s, q_head_local_idx);

                // --- Reduction Pipeline ---
                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    warp::wait(O_partial_arrived(s, q_head_local_idx, stage), (i / NUM_STAGES) % 2);

                    o_sv &O_smem = get_O_partial_smem(s, q_head_local_idx, stage);

                    // Load cur L_partial value
                    uint32_t src_ptr_L = static_cast<uint32_t>(__cvta_generic_to_shared(&L_smem.data[i]));
                    move<float>::lds(current_lse, src_ptr_L);
                    // Load O_partial_reg
                    warp::load(current_out, O_smem);

                    float max_lse = max(accumulated_lse, current_lse);

                    float accumulated_exp = exp2f(accumulated_lse - max_lse);
                    float current_exp = exp2f(current_lse - max_lse);

                    float new_denom = accumulated_exp + current_exp;

                    float accumulated_scale = accumulated_exp / new_denom;
                    float current_scale = current_exp / new_denom;

                    warp::mul(accumulated_out, accumulated_out, accumulated_scale);
                    warp::mul(current_out, current_out, current_scale);
                    warp::add(accumulated_out, accumulated_out, current_out);

                    // Update LSE accumulator:
                    accumulated_lse = max_lse + log2f(new_denom);

                    warp::arrive(O_partial_finished(s, q_head_local_idx, stage));
                }
                warp::arrive(L_partial_all_finished(s, q_head_local_idx));

                o_final_sv &O_final_smem = get_O_final_smem(s, q_head_local_idx);
                warp::store(O_final_smem, accumulated_out);
                warp::sync();

                warp::arrive(final_O_ready(s, q_head_local_idx));
                finish_shared_page(s);
            }
        }
    };

    // Storer Warp: Responsible for storing data from shared memory back to global memory.
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if (warp::laneid() < Q_HEADS_PER_INSTRUCTION) {
                int q_head_local_idx = warp::laneid();

                o_final_sv &O_final_smem = get_O_final_smem(s, q_head_local_idx);
                wait(final_O_ready(s, q_head_local_idx), 0);
                tma::store_async<cache_policy::NORMAL>(g.O_final, O_final_smem, {0, inst.q_head_start_idx + q_head_local_idx, 0, 0});
                tma::store_async_read_wait();
                finish_shared_page(s);

                atomicAdd(&g.barriers[{inst.layer_idx, GQA_REDUCTION_OPCODE, inst.q_head_start_idx + q_head_local_idx}], 1);
            }
            warp::sync();
         }
    };
};

// --- Python Bindings ---
#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(gqa_reduction, m) {
    m.doc() = "GQA Reduction VM Operation";
    kittens::py::bind_kernel<kvm<config, globals, rope_gqa_reduction_op<config>>>(m, "gqa_reduction",
        &globals::instructions,
        &globals::barriers,
        &globals::timings,
        &globals::L_partials,
        &globals::O_partials,
        &globals::O_final
    );
}