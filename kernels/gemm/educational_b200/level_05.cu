#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;
static constexpr int NUM_THREADS = WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

__global__
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int N
) {
    int col = blockIdx.x;
    int row = blockIdx.y;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile &As = al.allocate<a_tile>();
    b_tile &Bs = al.allocate<b_tile>();
    d_tile &Ds = al.allocate<d_tile>();

    __shared__ semaphore smem_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(smem_arrived, 0, 1);
    }
    __syncthreads();

    rt_bf<TILE_M, TILE_K> A_reg;
    rt_bf<TILE_K, TILE_N> B_reg;
    rt_bf<TILE_K, TILE_N, ducks::rt_layout::col> B_reg_col;
    rt_fl<TILE_M, TILE_N> C_accum;

    warp::zero(C_accum);
    int num_tiles = N / TILE_K;
    int phase = 0;

    for (int tile = 0; tile < num_tiles; ++tile) {
        if (threadIdx.x == 0) {
            tma::expect_bytes(smem_arrived, sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(As, A_layout, {row, tile}, smem_arrived);
            tma::load_async(Bs, B_layout, {tile, col}, smem_arrived);
        }

        wait(smem_arrived, phase);
        phase ^= 1;

        warp::load(A_reg, As);
        warp::load(B_reg, Bs);
        warp::swap_layout(B_reg_col, B_reg);
        warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);

        __syncthreads();
    }

    warp::store(Ds, C_accum);
    __syncthreads();

    if (threadIdx.x == 0) {
        tma::store_async(D_layout, Ds, {row, col});
        tma::store_async_read_wait();
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    a_gl A_layout{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl B_layout{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};

    dim3 blocks(N / TILE_N, N / TILE_M);
    int smem_size = MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    matmul_kernel<<<blocks, NUM_THREADS, smem_size>>>(A_layout, B_layout, D_layout, N);
}

#include "launch.cu"
