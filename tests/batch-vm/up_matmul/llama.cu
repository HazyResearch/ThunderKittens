#include "llama.cuh"

#include "up_matmul.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using up_matmul_op = up_matmul<default_config, llama_70b_globals>;

PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
                                 llama_70b_globals,
                                 up_matmul_op>>(m, "kvm_llama",
                                                  &llama_70b_globals::Bar,
                                                  &llama_70b_globals::instructions,
                                                  &llama_70b_globals::timings,
                                                  
                                                  &llama_70b_globals::up_weights, //
                                                  &llama_70b_globals::rms_gate_intermediates, // 
                                                  &llama_70b_globals::gate_silu_intermediates, //
                                                  &llama_70b_globals::silu_out // 
                                                  );
}

