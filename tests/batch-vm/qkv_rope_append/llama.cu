#include "llama.cuh"

#include "qkv_rope_append.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using qkv_rope_append_op = qkv_rope_append<default_config, llama_70b_globals>;

PYBIND11_MODULE(kvm_llama, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<kvm<default_config, 
                                 llama_70b_globals,
                                 qkv_rope_append_op>>(m, "kvm_llama",
                                                  &llama_70b_globals::Bar,
                                                  &llama_70b_globals::instructions,
                                                  &llama_70b_globals::timings,

                                                  &llama_70b_globals::qkv_weights, // 

                                                  &llama_70b_globals::k_cache, // 
                                                  &llama_70b_globals::v_cache, // 

                                                  &llama_70b_globals::rope_cos, // 
                                                  &llama_70b_globals::rope_sin, // 

                                                  &llama_70b_globals::rms_rope_intermediates, // 
                                                  &llama_70b_globals::q_post_rope, //
                                                  
                                                  &llama_70b_globals::routing_table, //

                                                  &llama_70b_globals::pos_id //
                                                  );
}