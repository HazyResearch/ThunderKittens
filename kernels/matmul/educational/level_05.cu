

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
#define NUM_WORKERS  (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B; 
    tile_gl C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    
    rt_fl<16,BLOCK_SIZE> C_accum;
    rt_fl<16,BLOCK_SIZE> C_accum_cpy;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int row = by; 
    int col = bx; 

    // int condition = (threadIdx.x == 0 && threadIdx.y == 0 & blockIdx.x == 0);

    kittens::warp::zero(C_accum_cpy);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        warpgroup::load(As, g.A, {0, 0, row, tile});
        warpgroup::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        warpgroup::mma_AB(C_accum, As, Bs);
        warpgroup::mma_async_wait();
        kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
        kittens::warp::zero(C_accum);
    }
    warpgroup::store(g.C, C_accum_cpy, {0, 0, row, col});
}

// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 

    // global pointers
    using a_gl = matmul_globals::tile_gl;
    using b_gl = matmul_globals::tile_gl; 
    using c_gl = matmul_globals::tile_gl;
    a_gl  a_arg{A, nullptr, nullptr, N, N};
    b_gl  b_arg{B, nullptr, nullptr, N, N};
    c_gl  c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // launch
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
