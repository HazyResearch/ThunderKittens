#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern void short_conv(
    torch::Tensor h,
    torch::Tensor x,
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for short conv"; // optional module docstring
    m.def("short_conv", short_conv);
}