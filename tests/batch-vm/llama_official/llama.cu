#include "llama.cuh"

#include "rms_norm.cu"
#include "qkv_rope_append.cu"
#include "attention_decode.cu"
#include "matmul_adds.cu"
#include "gate_silu.cu"
#include "up_matmul.cu"
#include "lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using attn_norm_op = attn_norm<llama_config, llama_8b_globals>;
using qkv_rope_append_op = qkv_rope_append<llama_config, llama_8b_globals>;
using attention_decode_op = attention_decode<llama_config, llama_8b_globals>;
using o_proj_op = o_proj<llama_config, llama_8b_globals>;

using mlp_norm_op = mlp_norm<llama_config, llama_8b_globals>;
using gate_silu_op = gate_silu<llama_config, llama_8b_globals>;
using up_matmul_op = up_matmul<llama_config, llama_8b_globals>;
using downproj_op = downproj<llama_config, llama_8b_globals>;

using lm_head_norm_op = lm_head_norm<llama_config, llama_8b_globals>;
using lm_head_op = lm_head<llama_config, llama_8b_globals>;


PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<llama_config, 
        llama_8b_globals,
        attn_norm_op,
        qkv_rope_append_op,
        attention_decode_op,
        o_proj_op,
        mlp_norm_op,
        gate_silu_op,
        up_matmul_op,
        downproj_op,
        lm_head_norm_op,
        lm_head_op
    >>(m, "kvm_llama",
        &llama_8b_globals::Bar,
        &llama_8b_globals::instructions,
        &llama_8b_globals::timings,

        &llama_8b_globals::qkv_weights,
        &llama_8b_globals::attn_norm_weights,
        &llama_8b_globals::o_weights,
        &llama_8b_globals::mlp_norm_weights,

        &llama_8b_globals::up_weights,
        &llama_8b_globals::gate_weights,
        &llama_8b_globals::down_weights,

        &llama_8b_globals::lm_head_norm_weights,
        &llama_8b_globals::lm_head_weights,

        &llama_8b_globals::k_cache,
        &llama_8b_globals::v_cache,

        &llama_8b_globals::rope_cos,
        &llama_8b_globals::rope_sin,

        &llama_8b_globals::hidden_states,
        &llama_8b_globals::rms_rope_intermediates,
        &llama_8b_globals::rms_gate_intermediates,

        &llama_8b_globals::silu_out,
        &llama_8b_globals::q_post_rope,
        &llama_8b_globals::attn_out,
        
        &llama_8b_globals::rms_lm_head_intermediates,
        &llama_8b_globals::logits,
        &llama_8b_globals::pos_id,
        &llama_8b_globals::attn_scale,
        &llama_8b_globals::rms_norm_eps,
        &llama_8b_globals::batch_size
    );
}
