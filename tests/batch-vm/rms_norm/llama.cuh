#pragma once

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

// Define MAX_SHARED_MEMORY if not already defined
#ifndef kittens_MAX_SHARED_MEMORY
#define kittens_MAX_SHARED_MEMORY 163840
#endif

#define OPCODE_RMS_NORM 1
#define OPCODE_QKV_RopeAppend 2
#define OPCODE_GQA_AttentionDecode 3
#define OPCODE_O_ProjResidual 4

#define OPCODE_POST_RMS_NORM 5
#define OPCODE_MLP_SiLU 6
#define OPCODE_MLP_Gate 7
#define OPCODE_DownProjResidual 8


#define LLAMA_NUM_LAYERS 32
#define LLAMA_HIDDEN_DIM 4096
#define LLAMA_INTERMEDIATE_DIM 14336
#define LLAMA_HEAD_DIM 128
#define LLAMA_NUM_ATTENTION_HEADS 32
#define LLAMA_NUM_KV_HEADS 8
#define LLAMA_KV_BLOCK_SIZE 16
#define LLAMA_MATMUL_OUT_BLOCK_SIZE 256
#define LLAMA_MATMUL_BATCH_BLOCK_SIZE 256

#define SM_COUNT 148
#define BATCH_SIZE 128

// timing event convention

#define TEVENT_LOADER_START 16
#define TEVENT_AT_GMEM_WAIT 17
#define TEVENT_DONE_GMEM_WAIT 18
#define TEVENT_LOADER_END 19

#define TEVENT_CONSUMER_START 24
#define TEVENT_CONSUMER_END 88

#define TEVENT_STORE_START 104
#define TEVENT_OUTPUT_READY 110
#define TEVENT_STORE_END 126

#define TEVENT_TRIPLES_START 100
#define TEVENT_TRIPLES_END 110
#define TEVENT_TRIPLES_STORE_START 124
#define TEVENT_TRIPLES_OUTPUT_READY 125

namespace kittens::prototype::vm
{
    // Add constant for MAX_SHARED_MEMORY
    constexpr int MAX_SHARED_MEMORY = kittens_MAX_SHARED_MEMORY;

    struct llama_config
    {
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
        static constexpr int NUM_CONSUMER_WARPS = 16;
        static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
        static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
        static constexpr int NUM_BLOCKS = 1;
        static constexpr int CLUSTER_BLOCKS = 1;
        static constexpr int MAX_SHARED_MEMORY = kittens::prototype::vm::MAX_SHARED_MEMORY;

        // Shared memory declared statically
        static constexpr int SCRATCH_BYTES = 8192+2048;
        static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
        static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

        // Shared memory declared dynamically
        static constexpr int PAGE_SIZE = 32768;
        static constexpr int NUM_PAGES = 6; // Hard-coded to 6 to avoid non-constant expression
        // static_assert(NUM_PAGES == 6, "NUM_PAGES must be 6");

        static constexpr bool TIMING_RECORD_ENABLED = false;

        static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

        static constexpr int CONSUMER_REGISTERS = 104;
        static constexpr int NON_CONSUMER_REGISTERS = 64;
    };

    template <int _hidden_dim, int _intermediate_dim, int _head_dim, int _num_attention_heads, int _num_kv_heads, int _kv_block_size, int _matmul_out_block_size, int _batch_size, int _sm_count>
    struct globals_t
    {

        constexpr static unsigned int matmul_batch_block_size = _matmul_out_block_size;
        constexpr static unsigned int kv_block_size = _kv_block_size;
        constexpr static unsigned int head_dim = _head_dim;
        constexpr static unsigned int hidden_dim = _hidden_dim;
        constexpr static unsigned int intermediate_dim = _intermediate_dim;
        constexpr static unsigned int num_attention_heads = _num_attention_heads;
        constexpr static unsigned int num_kv_heads = _num_kv_heads;
        constexpr static unsigned int sm_count = _sm_count;
        constexpr static unsigned int batch_size = _batch_size;

        using instruction_layout = ::kittens::prototype::vm::instruction_layout<llama_config>;
        using timing_layout = ::kittens::prototype::vm::timing_layout<llama_config>;

        using weights_t = gl<bf16, 1, -1, -1, hidden_dim, st_bf<matmul_batch_block_size, matmul_batch_block_size>, st_bf<128, 128>>; 
        using weights_big_indim_t = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<matmul_batch_block_size, matmul_batch_block_size>, st_bf<128, 128>>; 

        using activations_t = gl<bf16, 1, 1, batch_size, hidden_dim, sv_bf<hidden_dim>, sv_bf<head_dim>, sv_bf<16>, st_bf<64, 128>>;
        using activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim, sv_bf<intermediate_dim>, sv_bf<hidden_dim>, sv_bf<16>, st_bf<64, 128>>;
        using logits_t = gl<bf16, 1, 1, 1, -1, sv_bf<16>>;
        using norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>, sv_bf<16>>;
        using rope_table_t = gl<float, 1, 1, -1, head_dim, sv_fl<16>>;
        
        // FlashInfer Paged KV Cache Format: (max_num_pages, page_size, num_heads, head_dim)
        using kv_cache_t = gl<bf16, -1, 16, -1, head_dim, sv_bf<16>, tma::descriptor<st_bf<kv_block_size, head_dim>, 1>>;

        // num_layers by 6 ops per layer by up to 48 heads (Q + K + V)
        using barriers = gl<uint, 1, -1, 7, num_attention_heads + 2 * num_kv_heads>;

        // vm stuff
        barriers Bar;
        instruction_layout instructions;
        timing_layout timings;

        // model weights
        // weights_t qkv_weights;
        norm_weights_t attn_norm_weights;
        // weights_t o_weights;
        // norm_weights_t mlp_norm_weights;
        
        // weights_t up_weights; 
        // weights_t gate_weights;
        // weights_big_indim_t down_weights;

        // norm_weights_t lm_head_norm_weights;
        // weights_t lm_head_weights;

        // kv cache
        // kv_cache_t k_cache;
        // kv_cache_t v_cache;

        // other buffers
        // rope_table_t rope_cos;
        // rope_table_t rope_sin;

        // activation buffers
        activations_t hidden_states;
        activations_t rms_rope_intermediates;
        // activations_t rms_gate_intermediates;
        // activations_t gate_silu_intermediates;
        // activations_t q_post_rope;
        // activations_t attn_out;
        // activations_big_indim_t silu_out;
        // logits_t logits;

        // unsigned int pos_id;
        // float attn_scale;
        float rms_norm_eps;

        dim3 grid() { return dim3(sm_count); }
        dim3 block() { return dim3(llama_config::NUM_THREADS); }
        int dynamic_shared_memory() { return llama_config::DYNAMIC_SHARED_MEMORY; }
    };

    typedef globals_t<
        LLAMA_HIDDEN_DIM,
        LLAMA_INTERMEDIATE_DIM,
        LLAMA_HEAD_DIM,
        LLAMA_NUM_ATTENTION_HEADS,
        LLAMA_NUM_KV_HEADS,
        LLAMA_KV_BLOCK_SIZE,
        LLAMA_MATMUL_OUT_BLOCK_SIZE,
        BATCH_SIZE,
        SM_COUNT>
        llama_8b_globals;

    // Forward declarations with consistent parameter order
    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct post_rms_norm;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct qkv_rope_append;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct attention_decode;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct o_proj;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct pre_rms_norm;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct matmul_silu;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct matmul_gate;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct downproj;

    template <typename Config = llama_config, typename globals = llama_8b_globals>
    struct rms_lm_head;
}

