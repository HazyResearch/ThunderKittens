#include "../../../src/kittens.cuh"

// Number of elements in tile
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

// TODO can we run this w/ multiple warps?

using namespace kittens;

// (M, K) @ (K, N) = (M, N)
// here T has to be __half b/c we use bf_16 registers
template <typename T>
__global__ void simple_gemm(size_t m, size_t n, size_t k, T alpha,
                               T const* A, size_t lda, T const* B, size_t ldb,
                               T beta, T* C, size_t ldc) {
    // Here, each block contains a single warp, and that warp operates on an entire set of rows/columns of the matrix
    // And splits them into corresponding tiles
    // Number of tiles for row/column of matrix
    const size_t K_tiles = (k + MMA_K - 1) / MMA_K;
    // Row index of matrix A tile and col index of matrix B tile
    const size_t block_row = blockIdx.y * MMA_M;
    const size_t block_col = blockIdx.x * MMA_N;

    // c_reg is the accumulation register - progresive result of the tiles' matmul
    kittens::rt_bf_1x1<> c_reg;
    // Need to zero registers so we can accumulate all tiles into them
    kittens::zero(c_reg);

    #pragma unroll
    // TODO could we add more warps and reduce tile size?
    for (auto i = 0; i < K_tiles; i++) {
        kittens::rt_bf_1x1<> a_reg;
        // Load in B as column vector
        kittens::rt_bf_1x1<>::col_vec b_reg;
        // block_row*k: initial row offset, i*MMA_K describes which tile in that row
        kittens::load(a_reg, A + (block_row * k) + (i * MMA_K), k);
        // block_col: initial column offset, i*MMA_K*n describes stride of tiles across that column
        kittens::load(b_reg, B + block_col + (i * MMA_K * n), n);

        // Accumulate all tiles of row/column into c register after matmul
        mm_AB(c_reg, a_reg, b_reg, c_reg);
    }
    kittens::store(C + (block_row * n) + block_col, c_reg, n);

}

template<typename T>
void launch_simple_gemm(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, T const* B,
                                size_t ldb, T beta, T* C, size_t ldc){

    // 1 warp handles a single tile
    // Number of threads in each block is num warps times num threads per warp (32)
    const dim3 blockDim = NUM_WORKERS * kittens::WARP_THREADS;

    // Number of 16x16 tiles in the output matrix - one tile per block
    const dim3 gridDim = {
        (unsigned int) (n + MMA_N - 1) / MMA_N
        (unsigned int) (m + MMA_M - 1) / MMA_M);
    };

    simple_gemm<<blockDim, gridDim>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);


}

#include "harness.impl"