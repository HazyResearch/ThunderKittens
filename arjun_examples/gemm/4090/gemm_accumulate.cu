#include "../../../src/kittens.cuh"

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

// Number of warps per block
#define NUM_WORKERS = 16
// Number of blocks in each row of grid
#define NUM_BLOCKS = 16

using namespace kittens;

// TODO is this default layout for shared memory?
using layout = kittens::ducks::st_layout::swizzle;

// (M, K) @ (K, N) = (M, N)
// here T has to be __half b/c we use bf_16 registers
template <typename T>
__global__ void simple_gemm(size_t m, size_t n, size_t k, T alpha,
                               T const* A, size_t lda, T const* B, size_t ldb,
                               T beta, T* C, size_t ldc) {

    // TODO question for Ben: how would we do this w/ multiple warps - is the time taken to write to shared mem
    // for the accumulate operation not worth it?

    auto a_block_size = (m * k) / (gridDim.x * gridDim.y);
    auto b_block_size = (k * n) / (gridDim.x * gridDim.y);
    auto c_block_size = (m * n) / (gridDim.x * gridDim.y);
    // Single tile of a
    auto a_start = a_block_size * (blockIdx.y * gridDim.x + blockIdx.x);
    // Entire row of B
    auto b_start = b_block_size * (blockIdx.x * gridDim.y)
    auto c_start = blockDim.x * (m * n)

    // We need to know which warp we're in so we know what tile within the block to load
    auto warpid = kittens::warpid();
    const bf_16 *_a = A + a_block_start, *_b = B + b_block_start, *_c = C + block_start;

    // Used for when we load in columns from the second matrix
    st_bf_1x4<ducks::st_layout::swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();

    // 16x16 register tiles for portions of A and B (kittens is in multiple of 16),
    // Nothing inside v brackets b/c they are entire matrix tiles not row/column vectors
    // 1 register for each warp
    // c_reg is the output register - result of the matrix multiplication
    rt_bf_1x1<> a_reg, b_reg, c_reg;

    auto a_subtile_size = a_tile_size / NUM_WORKERS;
    auto a_subtile_size = b_tile_size / NUM_WORKERS;

    // For each block, we move that block's portion of A and B from global mem into registers
    // (so all warps can do ops on their respective subtiles)
    load(a_reg, _a + (num_a_subtiles * NUM_WORKERS + warpid) * a_reg.num_elements, a_reg.cols)
    load(b_reg, _b + (num_b_subtiles * NUM_WORKERS + warpid) * b_reg.num_elements, b_reg.cols)

    // What's the difference between "input" and "output" accumulator matrices in the docs:
    // https://github.com/HazyResearch/ThunderKittens/blob/2b9827dc11be408c386e70b84f923a11f70c7c33/src/ops/warp/register/tile/mma.cuh#L339

    mm_AB(c_reg, a_reg, b_reg, c_reg);
    // TODO is moving register->global faster than creating shared for c then moving all accumulated memory into global
    store(_c, c_reg, c_reg.cols);
}

template<typename T>
void launch_simple_gemm(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, T const* B,
                                size_t ldb, T beta, T* C, size_t ldc){

    // 1 warp handles a single tile
    // Number of threads in each block is num warps times num threads per warp (32)
    const dim3 blockDim = NUM_WORKERS * kittens::WARP_THREADS;

    const dim3 gridDim = {
        NUM_BLOCKS, NUM_BLOCKS
    };

    simple_gemm<<blockDim, gridDim>>(m, n, k);


}

#include "harness.impl"