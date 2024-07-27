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

#ifdef TK_COMPILE_ATTN_CAUSAL_INFERENCE
extern void attention_inference_forward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);
#endif

#ifdef TK_COMPILE_ATTN_CAUSAL_TRAINING
extern void attention_train_forward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l);
extern void attention_train_backward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg);
#endif

#ifdef TK_COMPILE_HEDGEHOG
extern void hedgehog_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor k_state, torch::Tensor kv_state,
    torch::Tensor q_map, torch::Tensor k_map,
    torch::Tensor alphas, torch::Tensor betas
);
#endif

#ifdef TK_COMPILE_BASED
extern void based_linear_prefill(
    int add_scale, int output_state,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor kv_a2, torch::Tensor kv_a1, torch::Tensor kv_a0
);
extern void based_step(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
            torch::Tensor kv_state, torch::Tensor k_state, torch::Tensor out);
extern void based_sliding_window(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Kernels"; // optional module docstring

#ifdef TK_COMPILE_ATTN_INFERENCE
    m.def("mha", attention_inference_forward, "Bidirectional forward MHA meant for inference. Takes device parameters Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Overwrites O.");
#endif

#ifdef TK_COMPILE_ATTN_TRAINING
    m.def("mha_train", attention_train_forward, "Bidirectional forward MHA meant for training. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("mha_train_backward", attention_train_backward, "Bidirectional backward MHA meant for training. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
#endif

#ifdef TK_COMPILE_ATTN_CAUSAL_INFERENCE
    m.def("mha_causal", attention_inference_forward_causal, "Causal forward MHA meant for inference. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64");
#endif

#ifdef TK_COMPILE_ATTN_CAUSAL_TRAINING
    m.def("mha_causal_train", attention_train_forward_causal, "Causal forward MHA meant for training. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("mha_causal_train_backward", attention_train_backward_causal, "Causal backward MHA meant for training. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
#endif

#ifdef TK_COMPILE_HEDGEHOG
    m.def("hedgehog", hedgehog_forward, """Hedgehog forward. Takes tensors (Q, K, V, O, k_state, kv_state, q_maps, k_maps, alphas, betas). Q,K,V,O are bf16 (B,H,N,128), q_maps and k_maps are bf16 (H,128,64), k_state is fp32 (B,H,128), kv_state is fp32 (B,H,128,128), and alphas and betas are fp32 (H,). Finally, N must be a multiple of 64.""");
#endif

#ifdef TK_COMPILE_BASED
    m.def("based_linear_prefill", based_linear_prefill, "Based linear prefill. Takes tensors (q, k, v, o, kv_a2, kv_a1, kv_a0). q, k, v, o are bf16 (B,H,N,128), kv_a2 is bf16 (B,H,256,64), kv_a1 is bf16 (B,H,16,64), and kv_a0 is bf16 (B,H,1,64). Finally, N must be a multiple of 64.");
    m.def("based_step", based_step, "Based step. Takes tensors (q, k, v, kv_state, k_state, out). q, k, v, out are bf16 (B,H,N,128), kv_state is bf16 (B,H,273,64), and k_state is bf16 (B,H,64).");
    m.def("based_sliding_window", based_sliding_window, "Based sliding window. Takes tensors (q, k, v, o). q, k, v, o are bf16 (B,H,N,64).");
#endif
}
 
