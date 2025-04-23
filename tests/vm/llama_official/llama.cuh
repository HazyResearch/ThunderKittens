#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>


// boo bad practice in a header file
using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using llama_config = default_config;

#define RMS_DoubleMatVecSiLU 1
#define DownProjResidual 2
#define RMS_QKV_MatVecRopeAppend 3
#define PartialAttention 4
#define AttentionReduction 5
#define O_ProjResidual 6

template <int hidden_dim, int intermediate_dim, int head_dim, int num_attention_heads, int num_kv_heads, int kv_block_size, int matvec_block_size, int sm_count>
struct llama_globals
{
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    using weights_t = gl<bf16, 1, -1, -1, hidden_dim, st_bf<16, 512>>;                 // assumed to be N by 2048 (X@W.T).
    using weights_big_indim_t = gl<bf16, 1, -1, -1, intermediate_dim, st_bf<16, 512>>; // assumed to be N by 2048 (X@W.T).

    using activations_t = gl<bf16, 1, 1, 1, hidden_dim, sv_bf<hidden_dim>>;
    using activations_big_indim_t = gl<bf16, 1, 1, 1, intermediate_dim, sv_bf<intermediate_dim>>;

    using norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>>;
    using rope_table_t = gl<bf16, 1, 1, -1, head_dim, sv_bf<head_dim>>;
    using kv_cache_t = gl<bf16, 1, -1, -1, head_dim, st_bf<kv_block_size, head_dim>>;
    
    // max attention partials == sm_count
    using attn_out_intermediates_t = gl<float, -1, sm_count, num_attention_heads, head_dim, sv_bf<head_dim>>;
    using attn_lse_intermediates_t = gl<float, -1, sm_count, num_attention_heads, head_dim, sv_bf<head_dim>>;

    // num_layers by 6 ops per layer by up to 32 heads.
    using barriers = gl<bf16, 1, -1, 6, num_attention_heads + 2 * num_kv_heads>;

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
    activations_t rms_scale;
    activations_t post_ln_rope_q;
    activations_t attn_out;
    attn_lse_intermediates_t attn_lse_intermediates;
    attn_out_intermediates_t attn_out_intermediates;
    activations_big_indim_t silu_out;

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
    constexpr static unsigned int sm_count = sm_count;
    

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};