#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using config = default_config;

#define RMS_QKV_MatVecRopeAppend 1
#define PartialAttention 2
#define AttentionReduction 3
#define DownProjResidual 4
#define O_ProjResidual 5
#define RMS_DoubleMatVecSiLU 6


template <int hidden_dim, int intermediate_dim, int head_dim, int num_attention_heads, int num_kv_heads, int kv_block_size, int matvec_block_size>
struct globals
{
    using output_tma = sv_bf<matvec_block_size>;
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using weights_normal_indim = gl<bf16, 1, -1, -1, hidden_dim, st_bf<16, 512>>;    // assumed to be N by 2048 (X@W.T).
    using weights_big_indim = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<16, 512>>; // assumed to be N by 2048 (X@W.T).
    using activations_normal_indim = gl<bf16, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>, output_tma>;
    using activations_big_indim = gl<bf16, 1, 1, 1, intermediate_dim, sv_bf<intermediate_dim>, output_tma>;
    using ln_weights = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>>;
    using rope_table = gl<bf16, 1, 1, -1, head_dim, sv_bf<head_dim>>;
    using kv_cache = gl<bf16, 1, -1, -1, head_dim, st_bf<kv_block_size, head_dim>, output_tma>;

    // TODO these are wrong
    using attn_out_intermediates = gl<float, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>, output_tma>;
    using attn_lse_intermediates = gl<float, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>, output_tma>;

    using barriers = gl<bf16, 1, -1, 6, 32>; // num_layers by 6 ops per layer by up to 32 heads.

    // model weights
    weights_normal_indim qkv_proj;
    ln_weights attn_ln_weight;
    weights_normal_indim o_proj;
    ln_weights mlp_ln_weight;
    weights_normal_indim up_proj;
    weights_normal_indim gate_proj;
    weights_big_indim down_proj;

    // other buffers
    rope_table rope_cos;
    rope_table rope_sin;

    // activation buffers
    activations_normal_indim rms_scale;
    activations_normal_indim post_ln_rope_q;
    activations_normal_indim attn_out;
    attn_lse_intermediates attn_lse_intermediates;
    attn_out_intermediates attn_out_intermediates;
    activations_big_indim silu_out;

    // barriers
    barriers Bar;
    instruction_layout instructions;
    timing_layout timings;

    unsigned int pos_id;
    float softmax_temp;
    float rms_norm_eps;

    constexpr static unsigned int matvec_block_size = matvec_block_size;
    constexpr static unsigned int kv_block_size = kv_block_size;
    constexpr static unsigned int head_dim = head_dim;
    constexpr static unsigned int hidden_dim = hidden_dim;
    constexpr static unsigned int intermediate_dim = intermediate_dim;
    constexpr static unsigned int num_attention_heads = num_attention_heads;
    constexpr static unsigned int num_kv_heads = num_kv_heads;

    unsigned int max_attn_partials;
    unsigned int max_barriers;
    unsigned int max_instructions;
    unsigned int max_timings;
    
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};