#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern void fused_rotary_tk(torch::Tensor x, torch::Tensor cos_in, torch::Tensor sin_in, torch::Tensor o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("fused_rotary_tk", fused_rotary_tk);
}
