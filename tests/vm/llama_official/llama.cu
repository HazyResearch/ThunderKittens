#include "llama.cuh"
#include "addmatvec.cuh"

#include "pyutils/pyutils.h"

PYBIND11_MODULE(matvec, m) {
    m.doc() = "matvec python module";
    kittens::py::bind_kernel<kvm<llama_config, llama_globals, MatvecOp<llama_config>>>(m, "matvec",
        &llama_globals::qkv_weights,
        &llama_globals::attn_ln_weights,
        &llama_globals::o_weights,
        &llama_globals::mlp_ln_weights,
        &llama_globals::up_weights,
        &llama_globals::gate_weights,
        &llama_globals::down_weights,
        &llama_globals::rope_cos,
        &llama_globals::rope_sin,
        &llama_globals::rms_scale,
        &llama_globals::post_ln_rope_q,
        &llama_globals::attn_out,
        &llama_globals::attn_lse_intermediates,
        &llama_globals::attn_out_intermediates,
        &llama_globals::silu_out,
        &llama_globals::Bar,
        &llama_globals::instructions,
        &llama_globals::timings,
        &llama_globals::pos_id,
        &llama_globals::softmax_temp,
        &llama_globals::rms_norm_eps
    );
}