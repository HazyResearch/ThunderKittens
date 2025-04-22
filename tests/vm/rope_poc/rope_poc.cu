#include <iostream>

#include "kittens.cuh"
#include "vm/vm.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int TEMP_OPCODE = 1; // this is not full instruction
constexpr int NUM_BLOCKS = 148;
constexpr int QKV_BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;

using qkv_rope_rv = rv_bf<16>;
using qkv_rope_sv = sv_bf<16>;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using qkv_proj_layout = gl<bf16, 1, 1, 1, (NUM_Q_HEADS + NUM_KV_HEADS * 2) * HEAD_DIM, qkv_rope_sv>;
    using rope_layout = gl<bf16, 1, 1, -1, HEAD_DIM, qkv_rope_sv>; // (N_max, D_h)
    using q_layout = gl<bf16, 1, 1, 1, NUM_Q_HEADS * HEAD_DIM, qkv_rope_sv>;
    using kv_layout = gl<bf16, -1, -1, NUM_KV_HEADS, HEAD_DIM, qkv_rope_sv>; // (L, N_max, H_kv, D_h)
    instruction_layout instructions;
    timing_layout timings;
    qkv_proj_layout QKV_proj;
    rope_layout rope_cos;
    rope_layout rope_sin;
    q_layout Q;
    kv_layout K_c;
    kv_layout V_c;
    int pos_id;
    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_op {
    static constexpr int opcode = TEMP_OPCODE;
    static constexpr int NUM_Q_ELEMS = NUM_Q_HEADS * HEAD_DIM;
    static constexpr int NUM_QK_ELEMS = NUM_Q_ELEMS + NUM_KV_HEADS * HEAD_DIM;
    static constexpr int Q_BLK_START = 0; // I like consistency
    static constexpr int K_BLK_START = NUM_Q_ELEMS / QKV_BLOCK_SIZE;
    static constexpr int V_BLK_START = NUM_QK_ELEMS / QKV_BLOCK_SIZE;

    struct parsed_instruction {
        int layer_idx;
        int qkv_block_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            qkv_block_idx = instruction[2]; // 16 elements per block!
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    // Semaphores
    __device__ static inline semaphore &setup_ready(state<config> &s) { return s.semaphores()[0]; }
    __device__ static inline semaphore &rope_cos_ready(state<config> &s) { return s.semaphores()[1]; }
    __device__ static inline semaphore &rope_sin_ready(state<config> &s) { return s.semaphores()[2]; }
    __device__ static inline semaphore &output_ready(state<config> &s) { return s.semaphores()[3]; }

    // Pages (very naive for now, no fine-grained usage)
    __device__ static inline int get_QKV_proj_page(state<config> &s) { return s.pid(0); }
    __device__ static inline int get_rope_cos_page(state<config> &s) { return s.pid(1); }
    __device__ static inline int get_rope_sin_page(state<config> &s) { return s.pid(2); }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            init_semaphore(setup_ready(s), 0, 1);
            init_semaphore(rope_cos_ready(s), 0, 1);
            init_semaphore(rope_sin_ready(s), 0, 1);
            init_semaphore(output_ready(s), 0, 1);
            return 4;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if (warp::laneid() >= 6 && warp::laneid() < 13)
                arrive(s.page_finished[s.pid(warp::laneid())], config::NUM_CONSUMER_WARPS); // Release the unused page immediately



            /* THIS PART DOESN'T EXIST IN FUSED OP */
            /* THIS PART DOESN'T EXIST IN FUSED OP */
            if (warp::laneid() == 0) {
                int qkv_proj_page_idx = get_QKV_proj_page(s);
                s.wait_page_ready(qkv_proj_page_idx);
                qkv_rope_sv &qkv_proj_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[qkv_proj_page_idx]);
                tma::expect(setup_ready(s), qkv_proj_smem);
                tma::load_async(qkv_proj_smem, g.QKV_proj, {0, 0, 0, inst.qkv_block_idx}, setup_ready(s));
                wait(setup_ready(s), 0);
            }
            warp::sync();
            /* THIS PART DOESN'T EXIST IN FUSED OP */
            /* THIS PART DOESN'T EXIST IN FUSED OP */



            // Load rotary encodings
            int rope_dim_idx = inst.qkv_block_idx % 4; // 0, 1, 2, 3
            if (warp::laneid() == 0) {
                int rope_cos_page_idx = get_rope_cos_page(s);
                s.wait_page_ready(rope_cos_page_idx);
                qkv_rope_sv &rope_cos_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[rope_cos_page_idx]);
                tma::expect(rope_cos_ready(s), rope_cos_smem);
                tma::load_async(rope_cos_smem, g.rope_cos, {0, 0, g.pos_id, rope_dim_idx}, rope_cos_ready(s));
            } else if (warp::laneid() == 1) {
                int rope_sin_page_idx = get_rope_sin_page(s);
                s.wait_page_ready(rope_sin_page_idx);
                qkv_rope_sv &rope_sin_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[rope_sin_page_idx]);
                tma::expect(rope_sin_ready(s), rope_sin_smem);
                tma::load_async(rope_sin_smem, g.rope_sin, {0, 0, g.pos_id, rope_dim_idx}, rope_sin_ready(s));
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            // Use a single warp
            if (warpid() == 0) {
                parsed_instruction inst{s};
    
                if (inst.qkv_block_idx < V_BLK_START) { // Q or K
                    int qkv_proj_page_idx = get_QKV_proj_page(s);
                    int rope_cos_page_idx = get_rope_cos_page(s);
                    int rope_sin_page_idx = get_rope_sin_page(s);
                    qkv_rope_sv &qkv_proj_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[qkv_proj_page_idx]);
                    qkv_rope_sv &rope_cos_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[rope_cos_page_idx]);
                    qkv_rope_sv &rope_sin_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[rope_sin_page_idx]);
                    qkv_rope_rv qkv_proj_reg;
                    qkv_rope_rv rope_cos_reg;
                    qkv_rope_rv rope_sin_reg;

                    // Load values
                    wait(rope_cos_ready(s), 0);
                    warp::load(qkv_proj_reg, qkv_proj_smem); // for this implementation only, we rely on rope_cos_ready (no need to wait in the fused op)
                    warp::load(rope_cos_reg, rope_cos_smem);
                    wait(rope_sin_ready(s), 0);
                    warp::load(rope_sin_reg, rope_sin_smem);

                    // Release pages
                    warp::arrive(s.page_finished[rope_cos_page_idx], config::NUM_CONSUMER_WARPS);
                    warp::arrive(s.page_finished[rope_sin_page_idx], config::NUM_CONSUMER_WARPS);
                    
                    // Fetch the neighbor values
                    int mod = (laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                    bf16 pair_val = __shfl_sync(MASK_ALL, qkv_proj_reg[0][0], laneid() + mod);
    
                    // Compute RoPE in-place
                    if (laneid() < 16)
                        qkv_proj_reg[0][0] = qkv_proj_reg[0][0] * rope_cos_reg[0][0] + bf16(-1 * mod) * pair_val * rope_sin_reg[0][0];
    
                    // Store the result in-place
                    warp::store(qkv_proj_smem, qkv_proj_reg);
                    warp::sync();
                } else { // V
                    wait(rope_cos_ready(s), 0); // delete this in fused op (no need to wait for projection, as it is already in reg/smem)
                }

                warp::arrive(output_ready(s));
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            if (warp::laneid() == 0) {
                parsed_instruction inst{s};

                int qkv_proj_page_idx = get_QKV_proj_page(s);
                qkv_rope_sv &qkv_proj_smem = reinterpret_cast<qkv_rope_sv &>(s.pages[qkv_proj_page_idx]);
                wait(output_ready(s), 0);
    
                if (inst.qkv_block_idx < K_BLK_START) { // Q
                    tma::store_async<cache_policy::NORMAL>(g.Q, qkv_proj_smem, {0, 0, 0, inst.qkv_block_idx});
                } else if (inst.qkv_block_idx < V_BLK_START) { // K
                    int base_index = (inst.qkv_block_idx - K_BLK_START) * QKV_BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / QKV_BLOCK_SIZE;
                    tma::store_async<cache_policy::NORMAL>(g.K_c, qkv_proj_smem, {inst.layer_idx, g.pos_id, head_idx, dim_idx});
                } else { // V
                    int base_index = (inst.qkv_block_idx - V_BLK_START) * QKV_BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / QKV_BLOCK_SIZE;
                    tma::store_async<cache_policy::NORMAL>(g.V_c, qkv_proj_smem, {inst.layer_idx, g.pos_id, head_idx, dim_idx});
                }

                tma::store_async_read_wait();
                arrive(s.page_finished[qkv_proj_page_idx], config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(rope_poc, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kvm<config, globals, rope_op<config>>>(m, "rope_poc",
        &globals::instructions,
        &globals::timings,
        &globals::QKV_proj,
        &globals::rope_cos,
        &globals::rope_sin,
        &globals::Q,
        &globals::K_c,
        &globals::V_c,
        &globals::pos_id
    );
}
