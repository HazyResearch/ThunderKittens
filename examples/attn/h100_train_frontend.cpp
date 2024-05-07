#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

extern void attention_train_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l);
extern void attention_train_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring

    m.def("attention_train_forward", attention_train_forward);
    m.def("attention_train_backward", attention_train_backward);
}
 
