#include "llama.cuh"

#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "rms_matvec_rope_append.cu"
#include "upgate.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

PYBIND11_MODULE(llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<config, llama_1b_globals,
                                 attention_partial<>,
                                 attention_reduction<>,
                                 rms_qkv_rope_append<>,
                                 downproj<>,
                                 o_proj<>,
                                 rms_upgate_silu<>>>(
        m, "llama",
        &llama_1b_globals::Bar,
        &llama_1b_globals::instructions,
        &llama_1b_globals::timings,

        &llama_1b_globals::qkv_weights,
        &llama_1b_globals::attn_norm_weights,
        &llama_1b_globals::o_weights,
        &llama_1b_globals::mlp_norm_weights,
        &llama_1b_globals::up_weights,
        &llama_1b_globals::gate_weights,
        &llama_1b_globals::down_weights,

        &llama_1b_globals::rope_cos,
        &llama_1b_globals::rope_sin,

        &llama_1b_globals::hidden_states,
        &llama_1b_globals::q_post_rope,
        &llama_1b_globals::attn_out,
        &llama_1b_globals::attn_lse_intermediates,
        &llama_1b_globals::attn_out_intermediates,
        &llama_1b_globals::silu_out,

        &llama_1b_globals::pos_id,
        &llama_1b_globals::attn_scale,
        &llama_1b_globals::rms_norm_eps);
}