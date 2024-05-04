#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern void fwd_train_attend_ker_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l);
extern void prep_train_attend_ker_tk(torch::Tensor o, torch::Tensor og, torch::Tensor d_vec); 
extern void bwd_train_attend_ker_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("fwd_train_attend_ker_tk", fwd_train_attend_ker_tk);
    m.def("prep_train_attend_ker_tk", prep_train_attend_ker_tk);
    m.def("bwd_train_attend_ker_tk", bwd_train_attend_ker_tk);
}
 
