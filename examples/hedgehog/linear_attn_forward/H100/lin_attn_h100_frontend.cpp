#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>


// extern void hh_lin_tk_exp(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor kv);
extern void hh_lin_tk_smd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o); 


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    // m.def("hh_lin_tk_exp", hh_lin_tk_exp);
    m.def("hh_lin_tk_smd", hh_lin_tk_smd);
}
 
