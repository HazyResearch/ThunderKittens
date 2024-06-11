#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern void attention_forward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    
    m.def("attention_forward_causal", attention_forward_causal);
}
 
