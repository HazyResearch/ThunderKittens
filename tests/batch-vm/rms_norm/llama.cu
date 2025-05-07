#include "llama.cuh"

#include "rms_norm.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using pre_rms_norm_op = pre_rms_norm<default_config, llama_70b_globals>;

using post_rms_norm_op = post_rms_norm<default_config, llama_70b_globals>;


PYBIND11_MODULE(kvm_llama_70b, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
                                 llama_70b_globals,
                                 pre_rms_norm_op
                                //  , post_rms_norm_op
                                 >>(m, "kvm_llama_70b",
                                                  &llama_70b_globals::Bar,
                                                  &llama_70b_globals::instructions,
                                                  &llama_70b_globals::timings,

                                                  &llama_70b_globals::attn_norm_weights,
                                                  &llama_70b_globals::hidden_states,
                                                  &llama_70b_globals::rms_rope_intermediates,

                                                //   &llama_70b_globals::mlp_norm_weights,
                                                //   &llama_70b_globals::hidden_states,
                                                //   &llama_70b_globals::rms_gate_intermediates,
                                                  &llama_70b_globals::rms_norm_eps
                                                  );
}

