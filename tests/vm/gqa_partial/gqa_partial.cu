#include <iostream>

#include "kittens.cuh"
#include "vm/vm.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

/*
    Instruction as defined on Python-side:

        class PartialAttention(Instruction):
            layer_idx: int
            kv_head_idx: int
            num_partials: int
            partial_idx: int

    Instruction format on CUDA:

        [0] = opcode (1)
        [1] = layer_idx
        [2] = kv_head_idx
        [3] = num_partials
        [4] = partial_idx

    Globals relevant to this instruction:

        - post_ln_rope_q (Tensor)         : (model_dim=2048,) - flattened from (num_heads=32, head_dim=64)
        - k_cache (Tensor)                : (num_layers=16, max_seq_len=131072, num_kv_heads=8, dim=64)
        - v_cache (Tensor)                : (num_layers=16, max_seq_len=131072, num_kv_heads=8, dim=64)
        - attn_lse_intermediates (Tensor) : (num_q_heads=32, max_attn_partials=1024)
        - attn_out_intermediates (Tensor) : (num_q_heads=32, max_attn_partials=1024, dim=64)
        - pos_id (int; pos of new token)  : # prev_tokens + 1
        - attn_kv_block_size (int)        : 16
        - num_attention_heads (int)       : 32
        - num_kv_heads (int)              : 8
        - softmax_temp (float)            : 1 / math.sqrt(dim=64)

    Llama Configs:

        >>> from transformers import LlamaConfig
        >>> LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        LlamaConfig {
            "architectures": [
                "LlamaForCausalLM"
            ],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ],
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.51.3",
            "use_cache": true,
            "vocab_size": 128256
        }
*/

constexpr int NUM_BLOCKS = 148;
constexpr int GQA_PARTIAL_OPCODE = 1;
constexpr int ATTN_BLOCK_SIZE = 16;
constexpr int GQA_RATIO = 4; // 32 : 8

using q_rt = rt_bf<16, 64>;                  // actual size is (G=4, d=64)
using q_st = st_bf<16, 64>;                  // actual size is (G=4, d=64) 2048B
using k_rt = rt_bf<16, 64>;                  // (ATTN_BLOCK_SIZE, d=64)
using v_rt = rt_bf<16, 64, col_l>;           // (ATTN_BLOCK_SIZE, d=64)
using kv_st = st_bf<16, 64>;                 // (ATTN_BLOCK_SIZE, d=64) 2048B
using attn_fl_rt = rt_fl<16, 16>;            // actual size is (G=4, ATTN_BLOCK_SIZE)
using attn_bf_rt = rt_bf<16, 16>;            // actual size is (G=4, ATTN_BLOCK_SIZE)
using max_vec_rv = col_vec<rt_fl<16, 64>>;   // actual size is (G=4)
using max_vec_sv = sv_fl<16>;                // actual size is (G=4)
using norm_vec_rv = col_vec<rt_fl<16, 64>>;  // actual size is (G=4)
using norm_vec_sv = sv_fl<16>;               // actual size is (G=4)
using l_rv = col_vec<rt_fl<16, 64>>;         // actual size is (G=4)
using l_sv = sv_fl<16>;                      // actual size is (G=4)
using o_rt = rt_fl<16, 64>;                  // actual size is (G=4, d=64)
using o_sv = sv_bf<64>;                      // (d=64)
using o_st = st_bf<16, 64>;                  // actual size is (G=4, d=64)

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using q_layout = gl<bf16, 1, 1, -1, -1, q_st>; // (H_q, D_h) = (32, 64)
    using kv_layout = gl<bf16, -1, -1, -1, -1, tma::descriptor<kv_st, 1>>; // (L, N_max, H_kv, D_h) = (16, 131072, 8, 64)
    using l_layout = gl<float, 1, 1, -1, -1, l_sv>; // (M_a, H_q) = (M_a, 32)
    using o_layout = gl<bf16, 1, -1, -1, -1, o_sv>; // (M_a, H_q, D_h) = (M_a, 32, 64) TODO should we do float for partial?
    instruction_layout instructions;
    timing_layout timings;
    q_layout Q;
    kv_layout K_c;
    kv_layout V_c;
    l_layout L;
    o_layout O;
    int pos_id;
    float attn_scale;
    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_gqa_partial_op {
    static constexpr int opcode = GQA_PARTIAL_OPCODE;
    static constexpr int NUM_STAGES = 4;
    static constexpr int HALF_PAGE_SIZE = config::PAGE_SIZE / 2;

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            kv_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    // We have 32 dynamic semaphores total
    __device__ static inline semaphore &Q_arrived(state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline semaphore &O_arrived(state<config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline semaphore &L_arrived(state<config> &s) {
        return s.semaphores()[2];
    }
    __device__ static inline semaphore &K_arrived(state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2];
    }
    __device__ static inline semaphore &V_arrived(state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2 + 1];
    }
    __device__ static inline semaphore &K_finished(state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline semaphore &V_finished(state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    // 13 pages
    __device__ static inline int get_Q_page(state<config> &s) {
        return s.pid(0);
    }
    __device__ static inline int get_O_page(state<config> &s) {
        return s.pid(1);
    }
    __device__ static inline int get_L_page(state<config> &s) {
        return s.pid(2);
    }
    __device__ static inline int get_K_page(state<config> &s, int stage) {
        return s.pid(3 + stage * 2);
    }
    __device__ static inline int get_V_page(state<config> &s, int stage) {
        return s.pid(3 + stage * 2 + 1);
    }

    template<ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_4_rows(SV (&dst)[4], const RT &src, int row4idx/*= 0, 1, 2, or 3*/) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols, "dst length must match src cols.");

        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U = SV::dtype;
        using U2 = base_types::packing<U>::packed_type;

        uint32_t dst_ptr[4];
        dst_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[0].data[0]));
        dst_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[1].data[0]));
        dst_ptr[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[2].data[0]));
        dst_ptr[3] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[3].data[0]));
        
        int laneid = ::kittens::laneid();
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;

        if (row4idx % 2 == 0 && laneid < 16) { // rows 0~3 or 8~11
            if (row4idx / 2 == 0) { // rows 0~3
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]); // note 2, not 1
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
                }
            } else { // rows 8~11
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
                }
            }
        } else if (row4idx % 2 == 1 && laneid >= 16) { // rows 4~7 or 12~15
            if (row4idx / 2 == 0) { // rows 4~7
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]); // note 2, not 1
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
                }
            } else { // rows 12~15
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
                }
            }
        }
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            init_semaphore(Q_arrived(s), 0, 1);
            init_semaphore(O_arrived(s), 0, 1);
            init_semaphore(L_arrived(s), 0, 1);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i), 0, 1);
                init_semaphore(V_arrived(s, i), 0, 1);
                init_semaphore(K_finished(s, i), 0, 1);
                init_semaphore(V_finished(s, i), 0, 1);
            }
            return 3 + 4 * NUM_STAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            // TODO release unused pages immediately
            // TODO use lanes more efficiently
            if (laneid() == 0) {
                // Setup
                parsed_instruction inst{s};
                int q_head_tile_idx = (inst.kv_head_idx * GQA_RATIO) / q_rt::tile_size_row; // 0 or 1
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + ATTN_BLOCK_SIZE - 1) / ATTN_BLOCK_SIZE;  // TODO indivisble token length
                int blocks_per_partial = (total_attn_blocks + inst.num_partials - 1) / inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx = min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                int Q_page_pid = get_Q_page(s);
                q_st &Q_smem = *reinterpret_cast<q_st*>(s.pages[Q_page_pid].data);

                // Load Q once
                s.wait_page_ready(Q_page_pid);
                tma::expect(Q_arrived(s), Q_smem);
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(Q_smem, g.Q, {0, 0, q_head_tile_idx, 0}, Q_arrived(s));

                // Run the pipeline!
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    int K_page_pid = get_K_page(s, stage);
                    int V_page_pid = get_V_page(s, stage);
                    kv_st &K_smem = *reinterpret_cast<kv_st*>(s.pages[K_page_pid].data);
                    kv_st &V_smem = *reinterpret_cast<kv_st*>(s.pages[V_page_pid].data);

                    if (i < NUM_STAGES) {
                        s.wait_page_ready(K_page_pid);
                        s.wait_page_ready(V_page_pid);
                    } else {
                        wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    // TODO: different lanes for K V
                    tma::expect(K_arrived(s, stage), K_smem);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(K_smem, g.K_c, {inst.layer_idx, i + start_blk_idx, inst.kv_head_idx, 0}, K_arrived(s, stage));
                    tma::expect(V_arrived(s, stage), V_smem);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(V_smem, g.V_c, {inst.layer_idx, i + start_blk_idx, inst.kv_head_idx, 0}, V_arrived(s, stage));
                }
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            if (warpid() == 0) {
                // Setup
                parsed_instruction inst{s};
                int q_head_local_idx = ((inst.kv_head_idx * GQA_RATIO) % q_rt::tile_size_row) / 4;
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + ATTN_BLOCK_SIZE - 1) / ATTN_BLOCK_SIZE;  // TODO indivisble token length
                int blocks_per_partial = (total_attn_blocks + inst.num_partials - 1) / inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx = min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                float softmax_temp = g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))

                // Even more setup
                q_rt Q_reg;
                k_rt K_reg;
                v_rt V_reg;
                l_rv L_reg;
                o_rt O_reg;
                attn_fl_rt attn_fl_reg;
                attn_bf_rt attn_bf_reg;
                max_vec_rv max_vec_reg;
                max_vec_rv scaled_max_vec_reg;
                max_vec_rv last_scaled_max_vec_reg;
                max_vec_rv diff_scaled_max_vec_reg;
                norm_vec_rv norm_vec_reg;
                warp::neg_infty(max_vec_reg);
                warp::zero(last_scaled_max_vec_reg); // just not +-inf
                warp::zero(norm_vec_reg);
                warp::zero(O_reg);
                int Q_page_pid = get_Q_page(s);
                int O_page_pid = get_O_page(s);
                int L_page_pid = get_L_page(s);
                q_st &Q_smem = *reinterpret_cast<q_st*>(s.pages[Q_page_pid].data);
                o_sv (&O_smem)[4] = *reinterpret_cast<o_sv(*)[4]>(s.pages[O_page_pid].data);
                l_sv &L_smem = *reinterpret_cast<l_sv*>(s.pages[L_page_pid].data);

                // Load Q once
                wait(Q_arrived(s), 0);
                warp::load(Q_reg, Q_smem);

                // Run the pipeline!
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    int K_page_pid = get_K_page(s, stage);
                    int V_page_pid = get_V_page(s, stage);
                    kv_st &K_smem = *reinterpret_cast<kv_st*>(s.pages[K_page_pid].data);
                    kv_st &V_smem = *reinterpret_cast<kv_st*>(s.pages[V_page_pid].data);

                    // Perform Q @ K.T 
                    warp::zero(attn_fl_reg);
                    warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(K_reg, K_smem);
                    warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                    warp::arrive(K_finished(s, stage));

                    // Obtain maximums per row (which is per head)
                    warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg); // includes previous max

                    // Scale attention block and maximums by sqrt(D_h)
                    warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                    warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                    // Calculate softmax numerator
                    warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                    warp::exp2(attn_fl_reg, attn_fl_reg);

                    // Calculate softmax denominator
                    warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                    warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                    // Normalize and accumulate numerator (A @ V)
                    warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                    warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(V_reg, V_smem);
                    warp::copy(attn_bf_reg, attn_fl_reg); // Convert to bf16 to do matmul
                    warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                    warp::arrive(V_finished(s, stage));

                    // Normalize and accumulate demoniator
                    warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                    warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                    // Save for next iteration
                    warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                }

                // Finish
                warp::div_row(O_reg, O_reg, norm_vec_reg);
                warp::log2(L_reg, norm_vec_reg);
                warp::add(L_reg, L_reg, last_scaled_max_vec_reg); // now L_reg contains the LSE

                // Wait for the output pages to be ready
                s.wait_page_ready(O_page_pid);
                s.wait_page_ready(L_page_pid);

                // Store the results
                store_4_rows(O_smem, O_reg, q_head_local_idx);
                warp::sync();
                warp::arrive(O_arrived(s));
                warp::store(L_smem, L_reg);
                warp::sync();
                warp::arrive(L_arrived(s));

                // TODO: release pages
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            if (laneid() == 0) {
                // Setup
                parsed_instruction inst{s};
                int q_head_start_idx = inst.kv_head_idx * GQA_RATIO; // 0, 4, 8, 12, 16, 20, 24, 28
                int q_head_vec_start_idx = q_head_start_idx % 16;
                int O_page_pid = get_O_page(s);
                int L_page_pid = get_L_page(s);
                o_sv (&O_smem)[4] = *reinterpret_cast<o_sv(*)[4]>(s.pages[O_page_pid].data);
                l_sv &L_smem = *reinterpret_cast<l_sv*>(s.pages[L_page_pid].data);

                // Store partial attention output to global memory
                wait(O_arrived(s), 0);
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[0], {0, 0, q_head_start_idx + 0, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[1], {0, 0, q_head_start_idx + 1, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[2], {0, 0, q_head_start_idx + 2, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[3], {0, 0, q_head_start_idx + 3, 0});

                // Store LSE to global memory
                wait(L_arrived(s), 0);
                float4 tmp; // manual load and store (maybe do this in consumer?)
                uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&L_smem.data[q_head_vec_start_idx]));
                float *dst_ptr = (float*)&g.L.raw_ptr[0 + q_head_start_idx];
                asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w) : "r"(src_ptr));
                asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(tmp.x), "f"(tmp.y), "f"(tmp.z), "f"(tmp.w), "l"(dst_ptr));

                // Wait for the stores to finish reading from shared memory
                tma::store_async_read_wait();
                arrive(s.page_finished[O_page_pid], config::NUM_CONSUMER_WARPS);

                // TODO clean up other pages
            }
            warp::sync();
         }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(gqa_partial, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel<config, globals, rope_gqa_partial_op<config>>>(m, "gqa_partial",
        &globals::instructions,
        &globals::timings,
        &globals::Q,
        &globals::K_c,
        &globals::V_c,
        &globals::L,
        &globals::O,
        &globals::pos_id,
        &globals::attn_scale
    );
}
