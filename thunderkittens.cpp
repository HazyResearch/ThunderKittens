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

#ifdef TK_COMPILE_ATTN
extern std::vector<torch::Tensor> attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal
); 
extern std::vector<torch::Tensor> attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor l_vec, torch::Tensor og, 
    bool causal
);
#endif

#ifdef TK_COMPILE_HEDGEHOG
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hedgehog(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor q_map, torch::Tensor k_map,
    torch::Tensor alphas, torch::Tensor betas
);
#endif

#ifdef TK_COMPILE_BASED
extern std::tuple<torch::Tensor, torch::Tensor> based(
    const torch::Tensor q, 
    const torch::Tensor k, 
    const torch::Tensor v
);
#endif

#ifdef TK_COMPILE_CYLON
extern void cylon(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor o, torch::Tensor kv_state,
    torch::Tensor q_map, torch::Tensor k_map
);
extern void cylon_bwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor q_map, torch::Tensor k_map,
    torch::Tensor o_grad, torch::Tensor kv_state,
    torch::Tensor q_grad, torch::Tensor k_grad, torch::Tensor v_grad,
    torch::Tensor q_map_grad, torch::Tensor k_map_grad
);
#endif

#ifdef TK_COMPILE_FLUX
extern torch::Tensor fused_flux_linear_gate(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor gate,
    const torch::Tensor y
);
extern torch::Tensor fused_flux_linear_gelu(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias
);
#endif

#ifdef TK_COMPILE_FFTCONV
extern torch::Tensor fftconv(
    const torch::Tensor u_real,
    const torch::Tensor kf_real,
    const torch::Tensor kf_imag,
    const torch::Tensor f_real,
    const torch::Tensor f_imag,
    const torch::Tensor finv_real,
    const torch::Tensor finv_imag,
    const torch::Tensor tw_real,
    const torch::Tensor tw_imag,
    const torch::Tensor twinv_real,
    const torch::Tensor twinv_imag,
    int B,
    int H,
    int N,
    int N1
);
#endif

#ifdef TK_COMPILE_FUSED_ROTARY
extern torch::Tensor fused_rotary(
    const torch::Tensor x,
    const torch::Tensor cos_in, 
    const torch::Tensor sin_in
);
#endif

#ifdef TK_COMPILE_FUSED_LAYERNORM
extern std::tuple<torch::Tensor, torch::Tensor> fused_layernorm(
    const torch::Tensor x,
    const torch::Tensor residual,
    const torch::Tensor norm_weight,
    const torch::Tensor norm_bias,
    float dropout_p
);
#endif

#ifdef TK_COMPILE_MAMBA2
extern torch::Tensor mamba2(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v,
    const torch::Tensor a
);
#endif

#ifdef TK_COMPILE_FP8_GEMM
extern torch::Tensor fp8_gemm(
    const torch::Tensor a,
    const torch::Tensor b
);
#endif

#ifdef TK_COMPILE_SCALED_MATMUL
extern torch::Tensor scaled_matmul(
    const torch::Tensor a,
    const torch::Tensor b,
    const torch::Tensor scale_a,
    const torch::Tensor scale_b
);
#endif

#ifdef TK_COMPILE_GROUP_GEMM
extern torch::Tensor& group_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const torch::Tensor& index,
    torch::Tensor& c
);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Kernels"; // optional module docstring

#ifdef TK_COMPILE_ATTN
    m.def("mha_forward",  torch::wrap_pybind_function(attention_forward), "Bidirectional forward MHA. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("mha_backward", torch::wrap_pybind_function(attention_backward), "Bidirectional backward MHA. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
#endif

#ifdef TK_COMPILE_HEDGEHOG
    m.def("hedgehog", hedgehog, "Hedgehog forward. Takes tensors (q, k, v, q_map, k_map, alphas, betas). q, k, v are bf16 (B,H,N,64), q_map and k_map are bf16 (H,E,64,64), alphas and betas are fp32 (H,E). Returns (B,H,N,64) in bf16.");
#endif

#ifdef TK_COMPILE_BASED
    m.def("based", based, "Based forward. Takes tensors (q, k, v). q, k, v are bf16 (B,H,N,64). Returns (B,H,N,64) in bf16.");
#endif

#ifdef TK_COMPILE_CYLON
    m.def("cylon", cylon, """Cylon forward. Takes tensors (q, k, v, o, kv_state, q_map, k_map). q, k, v, o are bf16 (B,H,N,64), kv_state is fp32 (B,H,E,64,64), q_map and k_map are bf16 (H,E,64,64).""");
    m.def("cylon_bwd", cylon_bwd, "Cylon backward. Takes tensors (q, k, v, q_map, k_map, o_grad, kv_state, q_grad, k_grad, v_grad, q_map_grad, k_map_grad). q, k, v, o_grad are bf16 (B,H,N,64), q_map and k_map are bf16 (H,E,64,64), kv_state is fp32 (B,H,E,64,64). Outputs q_grad, k_grad, v_grad are fp32 (B,H,N,64) and q_map_grad, k_map_grad are fp32 (H,E,64,64).");
#endif

#ifdef TK_COMPILE_FLUX
    m.def("tk_flux_linear_gate", fused_flux_linear_gate, "Flux linear gate. Takes tensors (x, weight, bias, gate, y).  x is (B, H1), weight is (H2, H1), bias and gate are (H2), y is (B, H2). x, weight, bias, gate, y are bf16. Returns (B, H2) in bf16.");
    m.def("tk_flux_linear_gelu", fused_flux_linear_gelu, "Flux linear gelu. Takes tensors (x, weight, bias).  x is (B, H1), weight is (H2, H1), bias is (H2). x, weight, bias are bf16. Returns (B, H2) in bf16.");
#endif

#ifdef TK_COMPILE_FFTCONV
    m.def("fftconv", fftconv, "FFTConv TK. Takes tensors (u_real, kf_real, kf_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, B, H, N, N1). All tensors are bf16 except B, H, N, N1 which are ints. Returns (B, H, N, N1) in bf16.");
#endif

#ifdef TK_COMPILE_FUSED_ROTARY
    m.def("fused_rotary", fused_rotary, "Rotary TK. Takes tensors (x, cos_in, sin_in). All tensors are bf16. Returns (B, H, N, 128) in bf16.");
#endif

#ifdef TK_COMPILE_FUSED_LAYERNORM
    m.def("fused_layernorm", fused_layernorm, "LayerNorm TK. Takes tensors (x, residual, norm_weight, norm_bias, dropout_p). x, residual, norm_weight, norm_bias are bf16. dropout_p is float. Returns (B, H, N, 128) in bf16.");
#endif

#ifdef TK_COMPILE_MAMBA2
    m.def("mamba2", mamba2, "Mamba2 TK. Takes tensors (q, k, v, a). q, k, v tensors are bf16 and a is float.");
#endif

#ifdef TK_COMPILE_FP8_GEMM
    m.def("fp8_gemm", fp8_gemm, "FP8 GEMM TK. Takes tensors (a, b). Both tensors are bf16. Returns (B, H, N, 128) in bf16.");
#endif

#ifdef TK_COMPILE_SCALED_MATMUL
    m.def("scaled_matmul", scaled_matmul, "Scaled Matmul TK. Takes tensors (a, b, scale_a, scale_b). a, b are fp8e4m3, scale_a, scale_b are float. Returns (M, N) in float.");
#endif

#ifdef TK_COMPILE_GROUP_GEMM
    m.def("group_gemm", group_gemm, "Group GEMM TK. Takes tensors (a, b, scale_a, scale_b, index). a, b are fp8, scale_a, scale_b are float, index is int64. Returns (M, N) in bf16.");
#endif

}