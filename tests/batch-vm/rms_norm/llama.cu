#include "llama.cuh"

#include "rms_norm.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using pre_rms_norm_op = pre_rms_norm<llama_config, llama_8b_globals>;

using post_rms_norm_op = post_rms_norm<llama_config, llama_8b_globals>;


PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<llama_config, 
                                 llama_8b_globals,
                                 pre_rms_norm_op
                                //  , post_rms_norm_op
                                 >>(m, "kvm_llama",
                                                  &llama_8b_globals::Bar,
                                                  &llama_8b_globals::instructions,
                                                  &llama_8b_globals::timings,

                                                  &llama_8b_globals::attn_norm_weights,
                                                  &llama_8b_globals::hidden_states,
                                                  &llama_8b_globals::rms_rope_intermediates,

                                                //   &llama_70b_globals::mlp_norm_weights,
                                                //   &llama_70b_globals::hidden_states,
                                                //   &llama_70b_globals::rms_gate_intermediates,
                                                  &llama_8b_globals::rms_norm_eps
                                                  );
}

