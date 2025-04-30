#include "llama.cuh"

#include "gqa.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using attention_partial_op = attention_partial<default_config, llama_70b_globals>;

PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, llama_70b_globals,
                                 attention_partial_op
                                 >>(m, "kvm_llama",
                                    &llama_70b_globals::instructions,
                                    &llama_70b_globals::Bar,
                                    &llama_70b_globals::timings,
                                    &llama_70b_globals::k_cache,
                                    &llama_70b_globals::v_cache,
                                    &llama_70b_globals::q_post_rope,
                                    &llama_70b_globals::attn_out,
                                    // &llama_70b_globals::attn_lse_intermediates,
                                    // &llama_70b_globals::attn_out_intermediates,
                                    &llama_70b_globals::pos_id,
                                    &llama_70b_globals::attn_scale
                                );
}