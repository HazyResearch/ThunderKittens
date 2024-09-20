#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern torch::Tensor fused_flux_linear_gate(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor gate,
    const torch::Tensor y
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch bindings for Flux"; // optional module docstring
    m.def("tk_flux_linear_gate", fused_flux_linear_gate);
}