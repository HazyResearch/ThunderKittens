#include "llama.cuh"
#include <limits>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;


using l_partial_sv = sv_fl<16>;
using o_sv = sv_fl<globals::head_dim>;
using o_rv = rv_fl<globals::head_dim>;
using o_final_sv = sv_bf<globals::head_dim>;


template <typename config = config, typename globals = globals>
struct attention_reduction {
    static constexpr int opcode = OPCODE_AttentionReduction;
    static constexpr int opcode_partial = OPCODE_PartialAttention;
    static constexpr int NUM_STAGES = 3;

    // --- Instruction Parsing ---
    // Instruction format:
    // [0] = opcode 
    // [1] = layer_idx
    // [2] = num_partials
    struct parsed_instruction {
        int layer_idx;
        int num_partials;
        int q_head_idx;
        __device__ inline parsed_instruction(state<config> &s) {
            layer_idx = s.instruction()[1];
            num_partials = s.instruction()[2];
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
    static constexpr size_t partials_region_size = NUM_STAGES * (sizeof(l_partial_sv) + sizeof(o_sv));
    static constexpr size_t final_o_offset = partials_region_size;
    static_assert(partials_region_size + sizeof(o_final_sv) <= config::PAGE_SIZE,
                  "Required shared memory exceeds configured page size.");

    __device__ static inline l_partial_sv &get_L_partial_smem(state<config> &s, int stage) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        size_t offset = stage * (sizeof(l_partial_sv) + sizeof(o_sv));
        return *reinterpret_cast<l_partial_sv*>(base_ptr + offset);
    }
     __device__ static inline o_sv &get_O_partial_smem(state<config> &s, int stage) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        size_t offset = stage * (sizeof(l_partial_sv) + sizeof(o_sv)) + sizeof(l_partial_sv);
        return *reinterpret_cast<o_sv*>(base_ptr + offset);
    }
    __device__ static inline o_final_sv &get_O_final_smem(state<config> &s) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        return *reinterpret_cast<o_final_sv*>(base_ptr + final_o_offset);
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
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
            if (laneid >= 1 && laneid < config::NUM_PAGES) arrive(s.page_finished[s.pid(laneid)], config::NUM_CONSUMER_WARPS); 
            if (laneid == 0) {
                while (*(volatile int *)&g.Bar[{inst.layer_idx, opcode_partial, (inst.q_head_idx / 4) * 4}] < inst.num_partials) {
                    __nanosleep(20);
                }

                wait_shared_page(s);

                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    l_partial_sv &L_smem = get_L_partial_smem(s, stage);
                    o_sv &O_smem = get_O_partial_smem(s, stage);

                    if (i >= NUM_STAGES) {
                        int prev_phase = (i / NUM_STAGES - 1) % 2;
                        wait(L_partial_finished(s, stage), prev_phase);
                        wait(O_partial_finished(s, stage), prev_phase);
                    }

                    L_smem.data[0] = __ldg(&g.attn_lse_intermediates.raw_ptr[(inst.q_head_idx * g.attn_lse_intermediates.cols()) + i]);
                    if (warp::laneid()==0) arrive(L_partial_arrived(s, stage));

                    tma::expect(O_partial_arrived(s, stage), O_smem);
                    tma::load_async<cache_policy::EVICT_FIRST>(O_smem, g.attn_out_intermediates, {0, inst.q_head_idx, i, 0}, O_partial_arrived(s, stage));
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
                o_rv accumulated_out;
                float accumulated_lse = -INFINITY;

                o_rv current_out;
                float current_lse;

                warp::zero(accumulated_out);

                // --- Reduction Pipeline ---
                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    warp::wait(L_partial_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::wait(O_partial_arrived(s, stage), (i / NUM_STAGES) % 2);

                    l_partial_sv &L_smem = get_L_partial_smem(s, stage);
                    o_sv &O_smem = get_O_partial_smem(s, stage);

                    // Load L_partial_reg (single float)
                    uint32_t src_ptr_L = static_cast<uint32_t>(__cvta_generic_to_shared(&L_smem.data[0]));
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

                    warp::arrive(L_partial_finished(s, stage));
                    warp::arrive(O_partial_finished(s, stage));
                }

                o_final_sv &O_final_smem = get_O_final_smem(s);
                warp::store(O_final_smem, accumulated_out);
                warp::sync();

                warp::arrive(final_O_ready(s));
                finish_shared_page(s);
            }
        }
    };

    // Storer Warp: Responsible for storing data from shared memory back to global memory.
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if (warp::laneid() == 0) {
                o_final_sv &O_final_smem = get_O_final_smem(s);
                wait(final_O_ready(s), 0);
                tma::store_async<cache_policy::NORMAL>(g.attn_out, O_final_smem, {0, inst.q_head_idx, 0, 0});
                tma::store_async_read_wait();
                finish_shared_page(s);
                atomicAdd(&g.Bar[{inst.layer_idx, opcode, inst.q_head_idx}], 1);
            }
            warp::sync();
         }
    };
};


