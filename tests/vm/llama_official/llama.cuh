#pragma once

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

#define OPCODE_RMS_DoubleMatVecSiLU 1
#define OPCODE_DownProjResidual 2
#define OPCODE_RMS_QKV_MatVecRopeAppend 3
#define OPCODE_PartialAttention 4
#define OPCODE_AttentionReduction 5
#define OPCODE_O_ProjResidual 6

template <int hidden_dim, int intermediate_dim, int head_dim, int num_attention_heads, int num_kv_heads, int kv_block_size, int matvec_block_size, int sm_count>
struct llama_globals
#define LLAMA_1B_HIDDEN_DIM 2048
#define LLAMA_1B_INTERMEDIATE_DIM 8192
#define LLAMA_1B_HEAD_DIM 64
#define LLAMA_1B_NUM_ATTENTION_HEADS 32
#define LLAMA_1B_NUM_KV_HEADS 8
#define LLAMA_1B_KV_BLOCK_SIZE 32
#define LLAMA_1B_MATVEC_BLOCK_SIZE 16
#define SM_COUNT 148

namespace kittens
{
namespace prototype
{
namespace vm
{

using config = default_config;

template <int _hidden_dim, int _intermediate_dim, int _head_dim, int _num_attention_heads, int _num_kv_heads, int _kv_block_size, int _matvec_block_size, int _sm_count>
struct globals_t
{

    constexpr static unsigned int matvec_block_size = _matvec_block_size;
    constexpr static unsigned int kv_block_size = _kv_block_size;
    constexpr static unsigned int head_dim = _head_dim;
    constexpr static unsigned int hidden_dim = _hidden_dim;
    constexpr static unsigned int intermediate_dim = _intermediate_dim;
    constexpr static unsigned int num_attention_heads = _num_attention_heads;
    constexpr static unsigned int num_kv_heads = _num_kv_heads;
    constexpr static unsigned int sm_count = _sm_count;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    using weights_t = gl<bf16, 1, -1, -1, hidden_dim, st_bf<matvec_block_size, 512>>;                 // assumed to be N by 2048 (X@W.T).
    using weights_big_indim_t = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<matvec_block_size, 512>>; // assumed to be N by 2048 (X@W.T).

    using activations_t = gl<bf16, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>>;
    using activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim, sv_bf<intermediate_dim>>;

    using norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>>;
    using rope_table_t = gl<bf16, 1, 1, -1, head_dim, sv_bf<head_dim>>;
    using kv_cache_t = gl<bf16, 1, -1, -1, head_dim, st_bf<kv_block_size, head_dim>>;

    // max attention partials == sm_count
    using attn_out_intermediates_t = gl<float, -1, num_attention_heads, sm_count, head_dim, sv_fl<head_dim>>;
    using attn_lse_intermediates_t = gl<float, 1, -1, num_attention_heads, sm_count, sv_fl<16>>;

    // num_layers by 6 ops per layer by up to 32 heads.
    using barriers = gl<bf16, 1, -1, 6, num_attention_heads + 2 * num_kv_heads>;

    // vm stuff
    barriers Bar;
    instruction_layout instructions;
    timing_layout timings;

    // model weights
    weights_t qkv_weights;
    norm_weights_t attn_ln_weights;
    weights_t o_weights;
    norm_weights_t mlp_ln_weights;
    weights_t up_weights;
    weights_t gate_weights;
    weights_big_indim_t down_weights;

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

    unsigned int pos_id;
    float attn_scale;
    float rms_norm_eps;

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<
    LLAMA_1B_HIDDEN_DIM,
    LLAMA_1B_INTERMEDIATE_DIM,
    LLAMA_1B_HEAD_DIM,
    LLAMA_1B_NUM_ATTENTION_HEADS,
    LLAMA_1B_NUM_KV_HEADS,
    LLAMA_1B_KV_BLOCK_SIZE,
    LLAMA_1B_MATVEC_BLOCK_SIZE,
    SM_COUNT>
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

} // namespace vm
} // namespace prototype
} // namespace kittens
