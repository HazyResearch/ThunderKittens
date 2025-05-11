#include "llama.cuh"

#include "rms_norm.cu"
#include "qkv_rope_append.cu"
#include "attention_decode.cu"
#include "matmul_adds.cu"
#include "gate_silu.cu"
#include "up_matmul.cu"
// #include "lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using pre_rms_norm_op = pre_rms_norm<default_config, llama_70b_globals>;
using qkv_rope_append_op = qkv_rope_append<default_config, llama_70b_globals>;
using attention_decode_op = attention_decode<default_config, llama_70b_globals>;
using o_proj_op = o_proj<default_config, llama_70b_globals>;

using post_rms_norm_op = post_rms_norm<default_config, llama_70b_globals>;
using gate_silu_op = gate_silu<default_config, llama_70b_globals>;
using up_matmul_op = up_matmul<default_config, llama_70b_globals>;
using downproj_op = downproj<default_config, llama_70b_globals>;

// using lm_head_rms_norm_op = lm_head_rms_norm<default_config, llama_70b_globals>;
// using lm_head_op = lm_head<default_config, llama_70b_globals>;


PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
        llama_70b_globals,
        pre_rms_norm_op,
        qkv_rope_append_op,
        attention_decode_op,
        o_proj_op,
        post_rms_norm_op,
        gate_silu_op,
        up_matmul_op,
        downproj_op
        // lm_head_op
    >>(m, "kvm_llama",
        &llama_70b_globals::Bar,
        &llama_70b_globals::instructions,
        &llama_70b_globals::timings,
        &llama_70b_globals::qkv_weights,
        &llama_70b_globals::attn_norm_weights,
        &llama_70b_globals::o_weights,
        &llama_70b_globals::mlp_norm_weights,
        &llama_70b_globals::up_weights,
        &llama_70b_globals::gate_weights,
        &llama_70b_globals::down_weights,
        &llama_70b_globals::lm_head_norm_weights,
        &llama_70b_globals::lm_head_weights,
        &llama_70b_globals::k_cache,
        &llama_70b_globals::v_cache,
        &llama_70b_globals::rope_cos,
        &llama_70b_globals::rope_sin,
        &llama_70b_globals::hidden_states,
        &llama_70b_globals::rms_rope_intermediates,
        &llama_70b_globals::rms_gate_intermediates,
        &llama_70b_globals::gate_silu_intermediates,
        &llama_70b_globals::q_post_rope,
        &llama_70b_globals::attn_out,
        &llama_70b_globals::silu_out,
        &llama_70b_globals::logits,
        &llama_70b_globals::routing_table,
        &llama_70b_globals::pos_id,
        &llama_70b_globals::attn_scale,
        &llama_70b_globals::rms_norm_eps
    );
}
