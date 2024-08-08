#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern torch::Tensor fftconv_tk(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    uint padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for conv1d"; // optional module docstring
    m.def("fftconv_tk", fftconv_tk);
}