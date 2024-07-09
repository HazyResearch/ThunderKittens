#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

/*

HOW TO REGISTER YOUR OWN, CUSTOM SET OF KERNELS:

1. Decide on the identifier which will go in config.py. For example, "attn_inference" is the identifier for the first set below.
2. Add the identifier to the dict of sources in config.py.
3. Add the identifier to the list of kernels you want compiled.
4. The macro defined here, when that kernel is compiled, will be "TK_COMPILE_{IDENTIFIER_IN_ALL_CAPS}." You need to add two chunks to this file.
4a. the extern declaration at the top.
4b. the registration of the function into the module.

m.def("attention_inference_forward", attention_inference_forward);

*/

#ifdef TK_COMPILE_ATTN_INFERENCE
extern void attention_inference_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);
#endif
#ifdef TK_COMPILE_ATTN_TRAINING
extern void attention_train_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l);
extern void attention_train_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Kernels"; // optional module docstring

#ifdef TK_COMPILE_ATTN_INFERENCE
    m.def("attention_inference_forward", attention_inference_forward);
#endif
#ifdef TK_COMPILE_ATTN_TRAINING
    m.def("attention_train_forward", attention_train_forward);
    m.def("attention_train_backward", attention_train_backward);
#endif
}
 
