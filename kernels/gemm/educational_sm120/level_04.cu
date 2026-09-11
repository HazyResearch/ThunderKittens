#include "kittens.cuh"
using namespace kittens;

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int BLOCK_SIZE = 32;
static constexpr int NUM_WORKERS = 1;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

struct matmul_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();

    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
    rt_fl<BLOCK_SIZE, BLOCK_SIZE> C_accum;

    int col = blockIdx.x;
    int row = blockIdx.y;

    warp::zero(C_accum);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        warp::load(As, g.A, {0, 0, row, tile});
        warp::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        warp::load(A_reg, As);
        warp::load(B_reg, Bs);
        warp::swap_layout(B_reg_col, B_reg);
        __syncthreads();
        warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads();
    }
    warp::store(g.C, C_accum, {0, 0, row, col});
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    using a_gl = matmul_globals::tile_gl;
    using b_gl = matmul_globals::tile_gl;
    using c_gl = matmul_globals::tile_gl;
    size_t SZ = (size_t)N;
    a_gl a_arg{(bf16*)A, nullptr, nullptr, SZ, SZ};
    b_gl b_arg{(bf16*)B, nullptr, nullptr, SZ, SZ};
    c_gl c_arg{(bf16*)C, nullptr, nullptr, SZ, SZ};
    matmul_globals g{a_arg, b_arg, c_arg, N};

    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
