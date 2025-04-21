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

constexpr int GQA_PARTIAL_OPCODE = 1;
constexpr int NUM_BLOCKS = 148;
constexpr int ATTN_BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS;
static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4.");

using q_rt = rt_bf<16, HEAD_DIM>;                 // only 4 rows are used
using q_st = st_bf<16, HEAD_DIM>;                 // only 4 rows are used
using k_rt = rt_bf<ATTN_BLOCK_SIZE, HEAD_DIM>;
using v_rt = rt_bf<ATTN_BLOCK_SIZE, HEAD_DIM, col_l>;
using kv_st = st_bf<ATTN_BLOCK_SIZE, HEAD_DIM>;
using attn_fl_rt = rt_fl<16, ATTN_BLOCK_SIZE>;    // only 4 values are used
using attn_bf_rt = rt_bf<16, ATTN_BLOCK_SIZE>;    // only 4 values are used
using max_vec_rv = col_vec<rt_fl<16, HEAD_DIM>>;  // only 4 values are used
using max_vec_sv = sv_fl<16>;                     // only 4 values are used
using norm_vec_rv = col_vec<rt_fl<16, HEAD_DIM>>; // only 4 values are used
using norm_vec_sv = sv_fl<16>;                    // only 4 values are used
using l_rv = col_vec<rt_fl<16, HEAD_DIM>>;        // only 4 values are used
using l_sv = sv_fl<16>;                           // only 4 values are used
using o_rt = rt_fl<16, HEAD_DIM>;                 // only 4 rows are used
using o_sv = sv_bf<HEAD_DIM>;
using o_st = st_bf<16, HEAD_DIM>;                 // only 4 rows are used

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using q_layout = gl<bf16, 1, 1, NUM_Q_HEADS, HEAD_DIM, q_st>;
    using kv_layout = gl<bf16, -1, -1, NUM_KV_HEADS, HEAD_DIM, tma::descriptor<kv_st, 1>>; // (L, N_max, H_kv, D_h)
    using l_layout = gl<float, 1, 1, NUM_Q_HEADS, -1, l_sv>; // (H_q, max_partials)
    using o_layout = gl<bf16, 1, NUM_Q_HEADS, -1, HEAD_DIM, o_sv>; // (H_q, max_partials, D_h) should we do float for partial O?
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
    static_assert(NUM_STAGES <= 4, "Modify page allocation for KVs.");

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

    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;
    __device__ static inline void wait_QOL_page(state<config> &s) { s.wait_page_ready(s.pid(QOL_PAGE)); }
    __device__ static inline void wait_KV_page(state<config> &s) { s.wait_page_ready(s.pid(KV_PAGE)); }
    __device__ static inline void finish_QOL_page(state<config> &s) { 
        if (warp::laneid() == 0) arrive(s.page_finished[s.pid(QOL_PAGE)], config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_KV_page(state<config> &s) { 
        if (warp::laneid() == 0) arrive(s.page_finished[s.pid(KV_PAGE)], config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline q_st &get_Q_smem(state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<q_st*>(s.pages[pid].data);
    }
    __device__ static inline o_sv (&get_O_smem(state<config> &s))[4] {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<o_sv (*)[4]>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st)
        );
    }
    __device__ static inline l_sv &get_L_smem(state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<l_sv*>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st) + sizeof(o_sv) * 4
        );
    }
    __device__ static inline kv_st &get_K_smem(state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st*>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (stage * 2)
        );
    }
    __device__ static inline kv_st &get_V_smem(state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st*>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (1 + stage * 2)
        );
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
        
        int laneid = warp::laneid();
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
    template<ducks::rt::row_layout RT>
    __device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
        if(col_idx >= dst.cols) return;
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2);
                    const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2) + 1;
                    if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x = val; }
                    else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y = val; }
                    else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
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
            // Release unused pages
            int laneid = warp::laneid();
            if (laneid >= 2 && laneid < config::NUM_PAGES) arrive(s.page_finished[s.pid(laneid)], config::NUM_CONSUMER_WARPS); 
            if (laneid == 0) {
                // Setup
                parsed_instruction inst{s};
                int q_head_tile_idx = (inst.kv_head_idx * GQA_RATIO) / q_rt::tile_size_row; // 0 or 1
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + ATTN_BLOCK_SIZE - 1) / ATTN_BLOCK_SIZE;
                int blocks_per_partial = (total_attn_blocks + inst.num_partials - 1) / inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx = min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                q_st &Q_smem = get_Q_smem(s);

                // Load Q once
                wait_QOL_page(s);
                tma::expect(Q_arrived(s), Q_smem);
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(Q_smem, g.Q, {0, 0, q_head_tile_idx, 0}, Q_arrived(s));

                // Run the pipeline!
                wait_KV_page(s);
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    if (i >= NUM_STAGES) {
                        wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    tma::expect(K_arrived(s, stage), K_smem);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(K_smem, g.K_c, {inst.layer_idx, i + start_blk_idx, inst.kv_head_idx, 0}, K_arrived(s, stage));
                    tma::expect(V_arrived(s, stage), V_smem);
                    tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(V_smem, g.V_c, {inst.layer_idx, i + start_blk_idx, inst.kv_head_idx, 0}, V_arrived(s, stage));
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
            if (warpid() == 0) { // TODO: divide into multiple warps
                // Setup
                parsed_instruction inst{s};
                int q_head_local_idx = ((inst.kv_head_idx * GQA_RATIO) % q_rt::tile_size_row) / 4;
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + ATTN_BLOCK_SIZE - 1) / ATTN_BLOCK_SIZE;
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
                q_st &Q_smem = get_Q_smem(s);
                o_sv (&O_smem)[4] = get_O_smem(s);
                l_sv &L_smem = get_L_smem(s);

                // Load Q once
                wait(Q_arrived(s), 0);
                warp::load(Q_reg, Q_smem);

                // Run the pipeline!
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    // Perform Q @ K.T 
                    warp::zero(attn_fl_reg);
                    warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                    warp::load(K_reg, K_smem);
                    warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                    warp::arrive(K_finished(s, stage));

                    // Mask out invalid positions at the end
                    if ((i + start_blk_idx + 1) * ATTN_BLOCK_SIZE > seq_len)
                        right_fill(attn_fl_reg, attn_fl_reg, seq_len % ATTN_BLOCK_SIZE, -999999999999.f);

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
                finish_KV_page(s);
                if (start_blk_idx < end_blk_idx) {
                    warp::div_row(O_reg, O_reg, norm_vec_reg);
                    warp::log2(L_reg, norm_vec_reg);
                    warp::add(L_reg, L_reg, last_scaled_max_vec_reg); // now L_reg contains the LSE
                } else {
                    // Very edgy case where no blocks are processed.
                    // Make the math work out during attention reduction!
                    warp::neg_infty(L_reg);
                }

                // Store the results
                store_4_rows(O_smem, O_reg, q_head_local_idx);
                warp::sync();
                warp::arrive(O_arrived(s));
                warp::store(L_smem, L_reg);
                warp::sync();
                warp::arrive(L_arrived(s));
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO; // 0, 4, 8, 12, 16, 20, 24, 28
            int q_head_vec_start_idx = q_head_start_idx % 16;

            // Store partial attention output to global memory
            if (laneid == 0) {
                o_sv (&O_smem)[4] = get_O_smem(s);
                wait(O_arrived(s), 0);
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[0], {0, q_head_start_idx + 0, inst.partial_idx, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[1], {0, q_head_start_idx + 1, inst.partial_idx, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[2], {0, q_head_start_idx + 2, inst.partial_idx, 0});
                tma::store_async<cache_policy::NORMAL>(g.O, O_smem[3], {0, q_head_start_idx + 3, inst.partial_idx, 0});
            }

            // Store LSE to global memory
            if (laneid < GQA_RATIO) {
                l_sv &L_smem = get_L_smem(s);
                wait(L_arrived(s), 0);
                // Can't do anything fancy with writing 4 spread-out values.
                // We can do this in the consumer if we want to (without using smem)
                float tmp;
                uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&L_smem.data[q_head_vec_start_idx + laneid]));
                float *dst_ptr = (float*)&g.L.raw_ptr[(q_head_start_idx + laneid) * g.L.cols() + inst.partial_idx];
                asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(tmp) : "r"(src_ptr));
                asm volatile("st.global.f32 [%0], %1;\n" : : "l"(dst_ptr), "f"(tmp));
            }

            // Wait and finish
            if (laneid == 0) {
                tma::store_async_read_wait();
                finish_QOL_page(s);
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
