#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#ifdef TORCH_COMPILE

std::vector<torch::Tensor> ring_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal
) {
    std::cout << "Ring Attention Forward" << std::endl;
    std::cout << "Q: " << q.sizes() << std::endl;
    std::cout << "K: " << k.sizes() << std::endl;
    std::cout << "V: " << v.sizes() << std::endl;
    std::cout << "Causal: " << causal << std::endl;
    std::cout << "Q dtype: " << q.dtype() << std::endl;
    std::cout << "K dtype: " << k.dtype() << std::endl;
    std::cout << "V dtype: " << v.dtype() << std::endl;
    std::cout << "Q device: " << q.device() << std::endl;
    std::cout << "K device: " << k.device() << std::endl;
    std::cout << "V device: " << v.device() << std::endl;
    std::cout << "Q is contiguous: " << q.is_contiguous() << std::endl;
    std::cout << "K is contiguous: " << k.is_contiguous() << std::endl;
    std::cout << "V is contiguous: " << v.is_contiguous() << std::endl;
    std::cout << "Q is pinned: " << q.is_pinned() << std::endl;
    std::cout << "K is pinned: " << k.is_pinned() << std::endl;
    std::cout << "V is pinned: " << v.is_pinned() << std::endl;
    std::cout << "Q is on CUDA: " << q.is_cuda() << std::endl;
    std::cout << "K is on CUDA: " << k.is_cuda() << std::endl;
    std::cout << "V is on CUDA: " << v.is_cuda() << std::endl;
    std::cout << "Q is on CPU: " << q.is_cpu() << std::endl;
    std::cout << "K is on CPU: " << k.is_cpu() << std::endl;
    std::cout << "V is on CPU: " << v.is_cpu() << std::endl;

    return {q, k, v};
}
std::vector<torch::Tensor> ring_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor l_vec, torch::Tensor og, 
    bool causal
) {
    std::cout << "Ring Attention Backward" << std::endl;
    std::cout << "Q: " << q.sizes() << std::endl;
    std::cout << "K: " << k.sizes() << std::endl;
    std::cout << "V: " << v.sizes() << std::endl;
    std::cout << "O: " << o.sizes() << std::endl;
    std::cout << "L: " << l_vec.sizes() << std::endl;
    std::cout << "Og: " << og.sizes() << std::endl;
    std::cout << "Causal: " << causal << std::endl;
    std::cout << "Q dtype: " << q.dtype() << std::endl;
    std::cout << "K dtype: " << k.dtype() << std::endl;
    std::cout << "V dtype: " << v.dtype() << std::endl;
    std::cout << "O dtype: " << o.dtype() << std::endl;
    std::cout << "L dtype: " << l_vec.dtype() << std::endl;
    std::cout << "Og dtype: " << og.dtype() << std::endl;
    std::cout << "Q device: " << q.device() << std::endl;
    std::cout << "K device: " << k.device() << std::endl;
    std::cout << "V device: " << v.device() << std::endl;
    std::cout << "O device: " << o.device() << std::endl;
    std::cout << "L device: " << l_vec.device() << std::endl;
    std::cout << "Og device: " << og.device() << std::endl;

    return {q, k, v, o, l_vec, og};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Ring Attention Kernels";
    m.def("ring_mha_forward",  torch::wrap_pybind_function(ring_attention_forward), "Bidirectional forward MHA. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("ring_mha_backward", torch::wrap_pybind_function(ring_attention_backward), "Bidirectional backward MHA. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
}

#else

#endif
