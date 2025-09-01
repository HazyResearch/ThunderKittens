#include "llama.cuh"

// #include "rms_norm.cu"
// #include "qkv_rope_append.cu"
// #include "attention_decode.cu"
// #include "matmul_adds.cu"
#include "gate_silu.cu"
// #include "up_matmul.cu"
// #include "rms_lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using gate_silu_op = gate_silu<default_config, llama_70b_globals>;

PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
                                 llama_70b_globals,
                                 gate_silu_op>>(m, "kvm_llama",
                                                  &llama_70b_globals::Bar,
                                                  &llama_70b_globals::instructions,
                                                  &llama_70b_globals::timings,

                                                  &llama_70b_globals::gate_weights,
                                                  &llama_70b_globals::rms_gate_intermediates,
                                                  &llama_70b_globals::silu_out
                                                  );
}

