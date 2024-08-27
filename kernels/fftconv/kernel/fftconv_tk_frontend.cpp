#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern torch::Tensor fftconv_tk(
    const torch::Tensor u_real,
    const torch::Tensor u_imag,
    const torch::Tensor kf_real,
    const torch::Tensor kf_imag,
    const torch::Tensor f_real,
    const torch::Tensor f_imag,
    const torch::Tensor finv_real,
    const torch::Tensor finv_imag,
    const torch::Tensor tw_real,
    const torch::Tensor tw_imag,
    const torch::Tensor twinv_real,
    const torch::Tensor twinv_imag,
    int B,
    int H,
    int N,
    int N1
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch binding for fftconv_tk"; // optional module docstring
    m.def("fftconv_tk", fftconv_tk);
}