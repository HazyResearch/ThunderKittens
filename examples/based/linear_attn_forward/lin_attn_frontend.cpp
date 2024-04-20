#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>
// #include "/var/cr05_data/sim_data/code/release/ThunderKittens/src/kittens.cuh"
// #include "/var/cr05_data/sim_data/code/release/ThunderKittens/src/common/pyutils/torch_helpers.cuh"
// using namespace kittens;
// #define NUM_WORKERS (16) // hardcoded, don't change

extern void based_fwd_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("based_fwd_tk", based_fwd_tk);
}
 
