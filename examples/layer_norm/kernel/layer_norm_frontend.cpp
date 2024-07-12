#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern void fused_ln_tk(
    int has_residual,
    float dropout_p,
    torch::Tensor x,
    torch::Tensor residual, 
    torch::Tensor norm_weight, torch::Tensor norm_bias, 
    torch::Tensor o, torch::Tensor out_resid
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("fused_ln_tk", fused_ln_tk);
}
 