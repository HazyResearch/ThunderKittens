#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern void tk_cumulative_sum(int add_reg, torch::Tensor k, torch::Tensor k_state_a1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("tk_cumulative_sum", tk_cumulative_sum);
}
 
