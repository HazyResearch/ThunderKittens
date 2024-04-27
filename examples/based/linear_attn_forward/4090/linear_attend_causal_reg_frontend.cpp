#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// extern void a2_compute(const torch::Tensor q, const torch::Tensor k, const torch::Tensor v, torch::Tensor y);
extern void a012_compute(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    // m.def("a2_compute", a2_compute);
    m.def("a012_compute", a012_compute);
}
 