#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

extern torch::Tensor batch_gemv(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for batch_gemv"; 
    m.def("batch_gemv", &batch_gemv);
}
 