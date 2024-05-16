#include "../../../src/kittens.cuh"

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

using namespace kittens;

template <typename T>
__global__ void simple_gemm_tt(size_t m, size_t n, size_t k, T alpha,
                               T const* A, size_t lda, T const* B, size_t ldb,
                               T beta, T* C, size_t ldc) {
    const size_t K_tiles = (k + MMA_K - 1) / MMA_K;
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= m && warp_col >= n) {
        return;
    }

    kittens::rt_hf_1x1<> cr;
    kittens::zero(cr);

    #pragma unroll
    for (size_t i = 0; i < K_tiles; i++) {
        kittens::rt_hf_1x1<> ar;
        kittens::rt<__half2, 1, 1, kittens::ducks::rt_layout::col> br;

        kittens::load(ar, A + warp_row * k + i * MMA_K, k);
        kittens::load(br, B + i * MMA_K * n + warp_col, n);

        kittens::mma_AB(cr, ar, br, cr);
    }
    kittens::store(C + warp_row * n + warp_col, cr, n);
}

template <typename T>
void launch_simple_gemm_tt(size_t m, size_t n, size_t k, T const* alpha,
                           T const* A, size_t lda, T const* B, size_t ldb,
                           T const* beta, T* C, size_t ldc,
                           cudaStream_t stream) {
    const dim3 block_dim{32u};
    const dim3 grid_dim{(unsigned int)(n + MMA_N - 1) / MMA_N,
                        (unsigned int)(m + MMA_M - 1) / MMA_M};
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

#include "harness.impl"
