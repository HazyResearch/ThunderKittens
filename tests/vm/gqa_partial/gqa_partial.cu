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
constexpr int ROPE_GQA_PARTIAL_OPCODE = 1;

using q_rt = rt_bf<16, 64>;        // actual size is (G=4, d=64)
using q_st = st_bf<16, 64>;        // actual size is (G=4, d=64)
using k_rt = rt_bf<16, 64>;        // (blk_len=16, d=64)
using v_rt = rt_bf<16, 64, col_l>; // (blk_len=16, d=64)
using kv_st = st_bf<16, 64>;       // (blk_len=16, d=64)
using attn_fl_rt = rt_fl<16, 16>;  // actual size is (G=4, blk_len=16)
using attn_bf_rt = rt_bf<16, 16>;  // actual size is (G=4, blk_len=16)
using l_rt = rv_bf<16>;            // actual size is (G=4)
using l_st = sv_bf<16>;            // actual size is (G=4)
using o_rt = rt_fl<16, 64>;        // actual size is (G=4, d=64)
using o_st = st_bf<16, 64>;        // actual size is (G=4, d=64)

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using q_layout = gl<bf16, 1, 1, -1, -1, q_st>; // (H_q, D_h) = (32, 64)
    using kv_layout = gl<bf16, -1, -1, -1, -1, tma::descriptor<kv_st, 1>>; // (L, N_max, H_kv, D_h) = (16, 131072, 8, 64)
    using l_layout = gl<bf16, 1, 1, -1, -1, l_st>; // (M_a, H_q) = (M_a, 32)
    using o_layout = gl<bf16, 1, -1, -1, -1, o_st>; // (M_a, H_q, D_h) = (M_a, 32, 64)
    instruction_layout instructions;
    timing_layout timings;
    q_layout Q;
    kv_layout K_c, V_c;
    l_layout L;
    o_layout O;
    int pos_id;
    int attn_blk_size;
    float softmax_temp;
    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct rope_gqa_partial_op {
    static constexpr int opcode = ROPE_GQA_PARTIAL_OPCODE;
    static constexpr int PIPELINE_STAGES = 3;

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

    __device__ static inline semaphore &inputs_arrived(state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[1];
    }

    template<ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_first_4_rows(SV (&dst)[4], const RT &src) {
        static_assert(sizeof(typename SV::dtype) == 2 && sizeof(typename RT::dtype) == 2, "Only 16-bit types are supported for now.");
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
        int row_idx = laneid / 4;
        int _col_idx = laneid % 4;

        if (laneid < 16) { // thread 16 starts at row 5
            for (int j = 0; j < src.width; j++) {
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]); // note 2, not 1
                int col_idx = _col_idx * 2 + j * 16;
                move<U2>::sts(dst_ptr[row_idx] + sizeof(typename SV::dtype)*col_idx, tmp[0]);
                move<U2>::sts(dst_ptr[row_idx] + sizeof(typename SV::dtype)*(col_idx+8), tmp[1]);
            }
        }
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            init_semaphore(inputs_arrived(s), 0, 1);
            init_semaphore(outputs_arrived(s), 0, 1);
            return 2;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            if (laneid() == 0) {
                int Q_page_pid = s.pid(0);
                int K_page_pid = s.pid(1);
                int V_page_pid = s.pid(2);
                s.wait_page_ready(Q_page_pid);
                s.wait_page_ready(K_page_pid);
                s.wait_page_ready(V_page_pid);
                q_st  &Q_smem = *reinterpret_cast<q_st*>(s.pages[Q_page_pid].data);
                kv_st &K_smem = *reinterpret_cast<kv_st*>(s.pages[K_page_pid].data);
                kv_st &V_smem = *reinterpret_cast<kv_st*>(s.pages[V_page_pid].data);
                tma::expect_bytes(inputs_arrived(s), sizeof(Q_smem) + sizeof(K_smem) + sizeof(V_smem));
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(Q_smem, g.Q, {0, 0, 0, 0}, inputs_arrived(s));
                tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(K_smem, g.K_c, {inst.layer_idx, 0, 0, 0}, inputs_arrived(s));
                tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(V_smem, g.V_c, {inst.layer_idx, 0, 0, 0}, inputs_arrived(s));
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            if (warpid() == 0) {
                // Wait for the inputs to be ready
                wait(inputs_arrived(s), 0);
    
                // Declare registers and shared memory
                int Q_page_pid = s.pid(0);
                int K_page_pid = s.pid(1);
                int V_page_pid = s.pid(2);
                int O_page_pid = s.pid(3);
                s.wait_page_ready(O_page_pid);
                q_st  &Q_smem = *reinterpret_cast<q_st*>(s.pages[Q_page_pid].data);
                kv_st &K_smem = *reinterpret_cast<kv_st*>(s.pages[K_page_pid].data);
                kv_st &V_smem = *reinterpret_cast<kv_st*>(s.pages[V_page_pid].data);
                o_st  &O_smem = *reinterpret_cast<o_st*>(s.pages[O_page_pid].data);
                q_rt Q_reg;
                k_rt K_reg;
                v_rt V_reg;
                o_rt O_reg;
                attn_fl_rt attn_fl_reg;
                attn_bf_rt attn_bf_reg;

                // Perform Q @ K.T 
                warp::load(Q_reg, Q_smem);
                warp::load(K_reg, K_smem);
                warp::zero(attn_fl_reg);
                warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);

                // Perform A @ V
                warp::load(V_reg, V_smem);
                attn_bf_reg = attn_fl_reg;
                warp::zero(O_reg);
                warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);

                // Let's figure out register layout
                // if (laneid() == 0) {
                //     printf("O_reg rows %d cols %d\n", O_reg.rows, O_reg.cols);
                //     printf("O_reg height %d width %d\n", O_reg.height, O_reg.width);
                //     printf("O_reg base tile packed per thread %d\n", O_reg.tiles[0][0].packed_per_thread);
                // }
                // for (int i = 0; i < O_reg.height; i++) {
                //     for (int j = 0; j < O_reg.width; j++) {
                //         for (int k = 0; k < O_reg.tiles[i][j].packed_per_thread; k++) {
                //             float2 value = O_reg.tiles[i][j].data[k];
                //             printf("tid %d: tiles[%d][%d].data[%d] = %f, %f\n", 
                //                 threadIdx.x, i, j, k, value.x, value.y);
                //             __syncwarp();
                //         }
                //     }
                // }

                // Store the results
                warp::store(O_smem, O_reg);
                warp::sync();

                // Arrive at semaphores
                warp::arrive(outputs_arrived(s));
                warp::arrive(s.page_finished[Q_page_pid], config::NUM_CONSUMER_WARPS);
                warp::arrive(s.page_finished[K_page_pid], config::NUM_CONSUMER_WARPS);
                warp::arrive(s.page_finished[V_page_pid], config::NUM_CONSUMER_WARPS);
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            if (laneid() == 0) {
                // Wait for the outputs to be ready
                wait(outputs_arrived(s), 0);

                // Declare registers and shared memory
                int O_page_pid = s.pid(3);
                o_st &O_smem = *reinterpret_cast<o_st*>(s.pages[O_page_pid].data);
                
                // Store to global memory
                tma::store_async(g.O, O_smem, {0, 0, 0, 0});
                tma::store_async_read_wait();

                // Arrive at semaphore
                arrive(s.page_finished[O_page_pid], config::NUM_CONSUMER_WARPS);
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
        &globals::attn_blk_size,
        &globals::softmax_temp
    );
}
