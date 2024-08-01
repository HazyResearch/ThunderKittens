#include "../../../src/kittens.cuh"


// Using Toeplitz matrix approach for short conv

using namespace kittens;

template <typename T>
__global__ void simple_gemm_tt(size_t d, T const* H, T const* u) {

}

template <typename T>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, T const* alpha,
                           T const* A, size_t lda, T const* B, size_t ldb,
                           T const* beta, T* C, size_t ldc,
                           cudaStream_t stream) {
    // TODO can we do more than 1 warp per block?
    const dim3 block_dim{
        (unsigned int)kittens::WARP_THREADS
    };
    const dim3 grid_dim{
        (unsigned int)(n + MMA_N - 1) / MMA_N,
        (unsigned int)(m + MMA_M - 1) / MMA_M
    };

    simple_gemm_tt<T><<<grid_dim, block_dim>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
}

// Explicit instantiation.
template void launch_simple_gemm_tt<__half>(size_t m, size_t n, size_t k,
                                            __half const* alpha,
                                            __half const* A, size_t lda,
                                            __half const* B, size_t ldb,
                                            __half const* beta, __half* C,
                                            size_t ldc, cudaStream_t stream);

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void fused_ln_tk(
    const torch::Tensor h,
    const torch::Tensor x,
    int K,
    int N
) {
    CHECK_INPUT(h);
    CHECK_INPUT(x);
    int batch = x.size(0);
    auto n    = x.size(1);

    TORCH_CHECK(batch == out.size(0) && batch == residual.size(0), "Differing batch sizes?");
    TORCH_CHECK(x.size(2) == d_model,           "x is d_model?");
    TORCH_CHECK(residual.size(2) == d_model,    "residual is d_model?");
    TORCH_CHECK(out.size(2) == d_model,         "out is d_model?");
    TORCH_CHECK(out_resid.size(2) == d_model,   "out_resid is d_model?");
    TORCH_CHECK(norm_weight.size(0) == d_model, "norm_weight is d_model?");
    TORCH_CHECK(norm_bias.size(0) == d_model,   "norm_bias is d_model?");

    TORCH_CHECK(x.size(1) % kittens::TILE_DIM == 0,        "sequence length is divisible by 16?");
    TORCH_CHECK(residual.size(1) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");
    TORCH_CHECK(out.size(1) % kittens::TILE_DIM == 0,      "sequence length is divisible by 16?");
    TORCH_CHECK(out_resid.size(1) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *h               = h.data_ptr<c10::BFloat16>();
    c10::BFloat16 *x               = x.data_ptr<c10::BFloat16>();

    const bf16* h_bf           = reinterpret_cast<const bf16*>(h_ptr);
    const bf16* x_bf           = reinterpret_cast<const bf16*>(x_ptr);

    // launch variables
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    // TODO change mem size
    unsigned long mem_size = d_model*NUM_WORKERS*2*2*2 + d_model*2*2;
    cudaFuncSetAttribute(
        fused_layer_norm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    int n_c = 2;
    int n_h = n/n_c;


    short_conv<<<batch*n_h,threads,mem_size>>>(
        h_bf, x_bf,
        K, N
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
#else
#include "harness.impl"
#endif
