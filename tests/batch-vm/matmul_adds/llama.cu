#include "llama.cuh"

#include "matmul_adds.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using o_proj_op = o_proj<default_config, llama_70b_globals>;
using downproj_op = downproj<default_config, llama_70b_globals>;

PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
                                 llama_70b_globals,
                                 o_proj_op
                                //  ,downproj_op
                                 >>(m, "kvm_llama",
                                                  &llama_70b_globals::Bar,
                                                  &llama_70b_globals::instructions,
                                                  &llama_70b_globals::timings,

                                                //   &llama_70b_globals::down_weights,
                                                //   &llama_70b_globals::silu_out,
                                                //   &llama_70b_globals::hidden_states,

                                                  &llama_70b_globals::o_weights,
                                                  &llama_70b_globals::hidden_states,
                                                  &llama_70b_globals::attn_out);
}

