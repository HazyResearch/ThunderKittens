#pragma once

#include "kittens.cuh"
#include "vm/vm.cuh"
#include "vm/util.cuh"
#include <iostream>

#define OPCODE_RMS_QKV_MatVecRopeAppend 1
#define OPCODE_PartialAttention 2
#define OPCODE_AttentionReduction 3
#define OPCODE_O_ProjResidual 4
#define OPCODE_RMS_DoubleMatVecSiLU 5
#define OPCODE_DownProjResidual 6
#define OPCODE_RMS_LM_Head 7

#define LLAMA_1B_NUM_LAYERS 16
#define LLAMA_1B_HIDDEN_DIM 2048
#define LLAMA_1B_INTERMEDIATE_DIM 8192
#define LLAMA_1B_HEAD_DIM 64
#define LLAMA_1B_NUM_ATTENTION_HEADS 32
#define LLAMA_1B_NUM_KV_HEADS 8
#define LLAMA_1B_KV_BLOCK_SIZE 16
#define LLAMA_1B_MATVEC_BLOCK_SIZE 16
#define LLAMA_1B_LM_HEAD_BLOCK_SIZE 32
#define LLAMA_1B_VOCAB_SIZE 128256
#define H100_SM_COUNT 132
#define B200_SM_COUNT 148

namespace kittens::prototype::vm {

    constexpr int ATOMIC_ADD_START = FREE_SLOTS_START;
    constexpr int ATOMIC_ADD_END = ATOMIC_ADD_START + 1;
    constexpr int EPILOGUE_START = ATOMIC_ADD_END + 1;
    constexpr int ACT_WAIT_DONE = EPILOGUE_START + 2;
    constexpr int WEIGHT_WAIT_START = ACT_WAIT_DONE + 3;
    constexpr int WEIGHT_WAIT_DONE = WEIGHT_WAIT_START + 4;
    constexpr int RMS_START = WEIGHT_WAIT_DONE + 4;
    constexpr int RMS_SCALE_WAIT_START = RMS_START + 1;
    constexpr int RMS_SCALE_WAIT_DONE = RMS_SCALE_WAIT_START + 1;
    constexpr int RMS_DONE = RMS_SCALE_WAIT_DONE + 1;

    constexpr int TEMP1 = RMS_DONE + 1;
    constexpr int TEMP2 = TEMP1 + 1;
    constexpr int TEMP3 = TEMP2 + 1;
    constexpr int TEMP4 = TEMP3 + 1;
    constexpr int TEMP5 = TEMP4 + 1;
    constexpr int TEMP6 = TEMP5 + 1;

    using config = default_config;

    template <int _num_layers, int _hidden_dim, int _intermediate_dim, int _head_dim, int _num_attention_heads, int _num_kv_heads, int _kv_block_size, int _matvec_block_size, int _sm_count>
    struct globals_t {

        // constexpr static unsigned int num_layers = _num_layers;
        // constexpr static unsigned int matvec_block_size = _matvec_block_size;
        // constexpr static unsigned int kv_block_size = _kv_block_size;
        // constexpr static unsigned int head_dim = _head_dim;
        // constexpr static unsigned int hidden_dim = _hidden_dim;
        // constexpr static unsigned int intermediate_dim = _intermediate_dim;
        // constexpr static unsigned int num_attention_heads = _num_attention_heads;
        // constexpr static unsigned int num_kv_heads = _num_kv_heads;
        // constexpr static unsigned int sm_count = _sm_count;

        constexpr static int num_layers = _num_layers;
        constexpr static int matvec_block_size = _matvec_block_size;
        constexpr static int kv_block_size = _kv_block_size;
        constexpr static int head_dim = _head_dim;
        constexpr static int hidden_dim = _hidden_dim;
        constexpr static int intermediate_dim = _intermediate_dim;
        constexpr static int num_attention_heads = _num_attention_heads;
        constexpr static int num_kv_heads = _num_kv_heads;
        constexpr static int sm_count = _sm_count;

        using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
        using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

        using weights_t = gl<bf16, 1, -1, -1, hidden_dim, st_bf<matvec_block_size, 512>>;                 // assumed to be N by 2048 (X@W.T).
        using weights_big_indim_t = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<matvec_block_size, 512>>; // assumed to be N by 2048 (X@W.T).

        using activations_t = gl<bf16, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>, sv_bf<head_dim>, sv_bf<matvec_block_size>>;
        using activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim, sv_bf<intermediate_dim>, sv_bf<hidden_dim>, sv_bf<matvec_block_size>>;
        using logits_t = gl<bf16, 1, 1, 1, -1, sv_bf<matvec_block_size>>;

        using norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>, sv_bf<matvec_block_size>>;
        using rope_table_t = gl<float, 1, 1, -1, head_dim, sv_fl<head_dim>>;
        using kv_cache_t = gl<bf16, -1, -1, -1, head_dim, sv_bf<matvec_block_size>, tma::descriptor<st_bf<kv_block_size, head_dim>, 1>>;

        // max attention partials == sm_count
        using attn_out_intermediates_t = gl<float, 1, num_attention_heads, -1, head_dim, sv_fl<head_dim>>;
        using attn_lse_intermediates_t = gl<float, 1, 1, num_attention_heads, -1, sv_fl<((sm_count + 15) / 16) * 16>>;

        // num_layers by 6 ops per layer by up to 48 heads (Q + K + V)
        using barriers = gl<uint, 1, -1, 6, num_attention_heads + 2 * num_kv_heads>;

        // vm stuff
        barriers Bar;
        instruction_layout instructions;
        timing_layout timings;

        // model weights
        weights_t qkv_weights;
        norm_weights_t attn_norm_weights;
        weights_t o_weights;
        norm_weights_t mlp_norm_weights;
        weights_t up_weights;
        weights_t gate_weights;
        weights_big_indim_t down_weights;
        norm_weights_t lm_head_norm_weights;
        weights_t lm_head_weights;
        // kv cache
        kv_cache_t k_cache;
        kv_cache_t v_cache;

        // other buffers
        rope_table_t rope_cos;
        rope_table_t rope_sin;

        // activation buffers
        activations_t hidden_states;
        activations_t q_post_rope;
        activations_t attn_out;
        attn_lse_intermediates_t attn_lse_intermediates;
        attn_out_intermediates_t attn_out_intermediates;
        activations_big_indim_t silu_out;
        logits_t logits;

        unsigned int pos_id;
        float attn_scale;
        float rms_norm_eps;
        bool skip_attn_reduction;

        dim3 grid() { return dim3(sm_count); }
        dim3 block() { return dim3(config::NUM_THREADS); }
        int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    };

    typedef globals_t<
        LLAMA_1B_NUM_LAYERS,
        LLAMA_1B_HIDDEN_DIM,
        LLAMA_1B_INTERMEDIATE_DIM,
        LLAMA_1B_HEAD_DIM,
        LLAMA_1B_NUM_ATTENTION_HEADS,
        LLAMA_1B_NUM_KV_HEADS,
        LLAMA_1B_KV_BLOCK_SIZE,
        LLAMA_1B_MATVEC_BLOCK_SIZE,
#ifndef KITTENS_BLACKWELL
        H100_SM_COUNT>
#else
        B200_SM_COUNT>
#endif
        llama_1b_globals;

    template <typename config = config, typename globals = llama_1b_globals>
    struct attention_partial;

    template <typename config = config, typename globals = llama_1b_globals>
    struct attention_reduction;

    template <typename config = config, typename globals = llama_1b_globals>
    struct rms_qkv_rope_append;

    template <typename config = config, typename globals = llama_1b_globals>
    struct downproj;

    template <typename config = config, typename globals = llama_1b_globals>
    struct o_proj;

    template <typename config = config, typename globals = llama_1b_globals>
    struct rms_upgate_silu;

}