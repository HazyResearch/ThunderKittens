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

// struct gqa_partial_op_types {
//     using q_st = st_bf<16, 64>;        // actual used is (group_size=4, dim=64)
//     using q_rt = rt_bf<16, 64>;        // actual used is (group_size=4, dim=64)
//     using k_rt = rt_bf<16, 64>;        // (block_seq_len=16, dim=64)
//     using v_rt = rt_bf<16, 64, col_l>; // (block_seq_len=16, dim=64)
//     using kv_st = st_bf<16, 64>;       // (block_seq_len=16, dim=64)
//     using afl_rt = rt_fl<16, 16>;      // actual used is (group_size=4, block_seq_len=16)
//     using abf_rt = rt_bf<16, 16>;      // actual used is (group_size=4, block_seq_len=16)
//     using l_rt = rv_bf<16>;            // actual used is (group_size=4)
//     using l_st = sv_bf<16>;            // actual used is (group_size=4)
//     using o_rt = rt_fl<16, 64>;        // actual used is (group_size=4, dim=64)
//     using o_st = st_bf<16, 64>;        // actual used is (group_size=4, dim=64)
// };
// using T = gqa_partial_op_types;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    // using q_in_layout     = gl<bf16, 1, 1, 1, 32 * 64, T::q_st>; // (1, 1, num_q_heads * head_dim) - as given
    // using q_layout        = gl<bf16, 1, 1, 32, 64, T::q_st>;     // (1, 1, num_q_heads, head_dim) - we convert to this
    // using kv_cache_layout = gl<bf16, -1, -1, -1, -1, T::kv_st>;  // (num_layers, max_seq_len, num_kv_heads, head_dim)
    // using l_layout        = gl<bf16, 1, 1, -1, -1, T::l_st>;     // (num_q_heads, max_attn_partials)
    // using o_layout        = gl<bf16, 1, -1, -1, -1, tma::descriptor<T::o_st, 1>>;    // (num_q_heads, max_attn_partials, head_dim)

    instruction_layout instructions;
    timing_layout timings;

    // q_in_layout post_ln_rope_q;
    // kv_cache_layout k_cache;
    // kv_cache_layout v_cache;
    // l_layout attn_lse_intermediates;
    // o_layout attn_out_intermediates;

    // int pos_id;              // N
    // int attn_kv_block_size;  // 16
    // int num_attention_heads; // 32
    // int num_kv_heads;        // 8
    // float softmax_temp;      // 1 / sqrt(head_dim)

    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct gqa_partial_op {
    static constexpr int opcode = GQA_PARTIAL_OPCODE;

    struct parsed_instruction {
        int opcode;
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
    };

    __device__ static inline semaphore &inputs_arrived(state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[1];
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            // init_semaphore(inputs_arrived(s), 0, 1);
            // init_semaphore(outputs_arrived(s), 0, 1);
            // return 2;
            return 0;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {

            // parsed_instruction inst {s.instruction()[0], s.instruction()[1], s.instruction()[2], s.instruction()[3], s.instruction()[4]};

            // if (laneid() == 0) {
                // int Q_page_pid = s.pid(0);
                // int K_page_pid = s.pid(1);
                // int V_page_pid = s.pid(2);
                // s.wait_page_ready(Q_page_pid);
                // s.wait_page_ready(K_page_pid);
                // s.wait_page_ready(V_page_pid);
                // T::q_st  &Q_smem = *reinterpret_cast<T::q_st*>(s.pages[Q_page_pid].data); // exactly 1 page (16KB)
                // T::kv_st &K_smem = *reinterpret_cast<T::kv_st*>(s.pages[K_page_pid].data);
                // T::kv_st &V_smem = *reinterpret_cast<T::kv_st*>(s.pages[V_page_pid].data);
                // tma::expect_bytes(inputs_arrived(s), sizeof(Q_smem) + sizeof(K_smem) + sizeof(V_smem));
                // tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(Q_smem, g.post_ln_rope_q, {0, 0, 0, 0}, inputs_arrived(s));
                // tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(K_smem, g.k_cache, {inst.layer_idx, 0, inst.kv_head_idx, 0}, inputs_arrived(s));
                // tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(V_smem, g.v_cache, {inst.layer_idx, 0, inst.kv_head_idx, 0}, inputs_arrived(s));
            // }

            // using q_layout        = gl<bf16, 1, 1, 1, -1, T::q_st>;     // (1, 1, 1, num_q_heads * head_dim)
            // using kv_cache_layout = gl<bf16, -1, -1, -1, -1, T::kv_st>; // (num_layers, max_seq_len, num_kv_heads, head_dim)
            // using l_layout        = gl<bf16, 1, 1, -1, -1, T::l_st>;    // (num_q_heads, max_attn_partials)
            // using o_layout        = gl<bf16, 1, -1, -1, -1, T::o_st>;   // (num_q_heads, max_attn_partials, head_dim)
        
            // struct gqa_partial_op_types {
            //     using q_rt = rt_bf<32, 64>;
            //     using kv_rt = rt_bf<32, 64>;
            //     using o_rt = rt_fl<32, 32>;
            //     using q_st = st_bf<32, 64>; // <= 16KB
            //     using kv_st = st_bf<32, 64>;
            //     using l_st = st_bf<32, 32>;
            //     using o_st = st_bf<32, 32>;
            // }
            // using gqa_partial_op_types = T;
            
            // using config = default_config;
            // struct globals {
            //     using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
            //     using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
            
            //     using q_layout        = gl<bf16, 1, 1, 1, -1, T::q_st>;     // (1, 1, 1, num_q_heads * head_dim)
            //     using kv_cache_layout = gl<bf16, -1, -1, -1, -1, T::kv_st>; // (num_layers, max_seq_len, num_kv_heads, head_dim)
            //     using l_layout        = gl<bf16, 1, 1, -1, -1, T::l_st>;    // (num_q_heads, max_attn_partials)
            //     using o_layout        = gl<bf16, 1, 1, -1, -1, T::o_st>;    // (num_q_heads, max_attn_partials)
            
            //     instruction_layout instructions;
            //     timing_layout timings;
            
            //     q_layout post_ln_rope_q;
            //     kv_cache_layout K_cache;
            //     kv_cache_layout V_cache;
            //     l_layout attn_lse_intermediates;
            //     l_layout attn_out_intermediates;

            // parsed_instruction inst = parse_instruction(g, s);



            // if(args.iter == 1) group<12>::sync(11); // wait for the consumer to finish its setup, before we do the second load.
            // if(warpgroup::warpid() == 0) {
            //     int pos = args.common.start_pos + NUM_ROWS*args.iter;
            //     int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
            //     int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, pos/PAGE_SIZE}];
            //     // next page we need to load?
            //     tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
            //     tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.kcache, args.globals.K_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
            //     tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.vcache, args.globals.V_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
            //     if(laneid() == 0) arrive(args.inputs_arrived, 3);
            // }
            // warpgroup::sync(5);

            // uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

            // int pipeline_stage = 0;
            // for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
            //     wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
            //     warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int Q_page = get_a_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(Q_page);
            //         }
            //         st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(a, g.A, {inst.row+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int b_page = get_b_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(b_page);
            //         }
            //         st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(b, g.B, {inst.col+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            // }
            // warp::sync();
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { 
            // if (!laneid()) {
            //     const char *msg = "launcher";
            //     printf("Block %d, Warp ID %d, Lane ID %d, opcode: %d: %s\n", blockIdx.x, warpid(), laneid(), s.instruction()[0], msg);
            // }

        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            // if (warpid() == 0) {
            //     // Wait for the inputs to be ready
            //     wait(inputs_arrived(s), 0);
    
            //     // Move from smem to registers
            //     int Q_page_pid = s.pid(0);
            //     int K_page_pid = s.pid(1);
            //     int V_page_pid = s.pid(2);
            //     int O_page_pid = s.pid(3);
            //     s.wait_page_ready(O_page_pid);
            //     T::q_st  &Q_smem = *reinterpret_cast<T::q_st*>(s.pages[Q_page_pid].data);
            //     T::kv_st &K_smem = *reinterpret_cast<T::kv_st*>(s.pages[K_page_pid].data);
            //     T::kv_st &V_smem = *reinterpret_cast<T::kv_st*>(s.pages[V_page_pid].data);
            //     T::o_st  &O_smem = *reinterpret_cast<T::o_st*>(s.pages[O_page_pid].data);
            //     T::q_rt Q_reg;
            //     T::k_rt K_reg;
            //     T::v_rt V_reg; // must use col layout
            //     T::afl_rt att_fl_reg;
            //     T::abf_rt att_bf_reg;
            //     T::o_rt O_reg;

            //     // warp::load(Q_reg, Q_smem);
            //     // warp::load(K_reg, K_smem);
            //     // warp::load(V_reg, V_smem);

            //     // warp::zero(att_fl_reg);
            //     // warp::mma_ABt(att_fl_reg, Q_reg, K_reg, att_fl_reg); // Q @ K.T

            //     // warp::zero(O_reg);
            //     // att_bf_reg = att_fl_reg; // convert to bf16 before matmul
            //     // warp::mma_AB(O_reg, att_bf_reg, V_reg, O_reg); // att @ V

            //     // warp::store(O_smem, O_reg);
            //     // warp::sync();
            //     warp::arrive(outputs_arrived(s));

            //     warp::arrive(s.page_finished[Q_page_pid], config::NUM_CONSUMER_WARPS);
            //     warp::arrive(s.page_finished[K_page_pid], config::NUM_CONSUMER_WARPS);
            //     warp::arrive(s.page_finished[V_page_pid], config::NUM_CONSUMER_WARPS);

            //     // if (laneid() == 0) {
            //     //     printf("K batch: %d\n", g.K_c.batch());
            //     //     printf("K head: %d\n", g.K_c.depth());
            //     //     printf("K row: %d\n", g.K_c.rows());
            //     //     printf("K col: %d\n", g.K_c.cols());

            //     //     float Q_smem_sum = 0;
            //     //     float Q_g_sum = 0;
            //     //     float Q_reg_sum = 0;
            //     //     for (int i = 0; i < 64 * 32; i++) {
            //     //         Q_g_sum += (float)g.Q.raw_ptr[i];
            //     //         Q_smem_sum += (float)Q_smem.data[i];
            //     //     }
            //     //     for (int i = 0; i < Q_reg.height; ++i) {
            //     //         for (int j = 0; j < Q_reg.width; ++j) {
            //     //             for (int k = 0; k < 128; ++k) {
            //     //                 Q_reg_sum += (float)*reinterpret_cast<bf16*>(&Q_reg.tiles[i][j].data[k]);
            //     //                 Q_reg_sum += (float)*(reinterpret_cast<bf16*>(&Q_reg.tiles[i][j].data[k]) + 1);
            //     //             }
            //     //         }
            //     //     }
            //     //     printf("Q_g_sum: %f\n", (float)(Q_g_sum));
            //     //     printf("Q_smem_sum: %f\n", (float)(Q_smem_sum));
            //     //     printf("Q_reg_sum: %f\n", (float)(Q_reg_sum));
                    
            //     //     float K_smem_sum = 0;
            //     //     float K_g_sum = 0;
            //     //     for (int i = 0; i < 64 * 32; i++) {
            //     //         K_g_sum += (float)g.K_c.raw_ptr[i];
            //     //         K_smem_sum += (float)K_smem.data[i];
            //     //     }
            //     //     printf("K_g_sum: %f\n", (float)(K_g_sum));
            //     //     printf("K_smem_sum: %f\n", (float)(K_smem_sum));
    
            //     //     float V_smem_sum = 0;
            //     //     float V_g_sum = 0;
            //     //     for (int i = 0; i < 64 * 32; i++) {
            //     //         V_g_sum += (float)g.V_c.raw_ptr[i];
            //     //         V_smem_sum += (float)V_smem.data[i];
            //     //     }
            //     //     printf("V_g_sum: %f\n", (float)(V_g_sum));
            //     //     printf("V_smem_sum: %f\n", (float)(V_smem_sum));
            //     // }
            // }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            // if (laneid() == 0) {
            //     wait(outputs_arrived(s), 0);
            //     int O_page_pid = s.pid(3);
            //     T::o_st &O_smem = *reinterpret_cast<T::o_st*>(s.pages[O_page_pid].data);
    
            //     // tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.attn_out_intermediates, O_smem, {0, 0, 0, 0});
            //     // tma::store_async_read_wait();
            //     arrive(s.page_finished[O_page_pid], config::NUM_CONSUMER_WARPS);
            // }
            // warp::sync();
         }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(gqa_partial, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel<config, globals, gqa_partial_op<config>>>(m, "gqa_partial",
        &globals::instructions,
        &globals::timings
        // &globals::post_ln_rope_q,
        // &globals::k_cache,
        // &globals::v_cache,
        // &globals::attn_lse_intermediates,
        // &globals::attn_out_intermediates
        // &globals::pos_id,
        // &globals::attn_kv_block_size,
        // &globals::num_attention_heads,
        // &globals::num_kv_heads,
        // &globals::softmax_temp
    );
}
