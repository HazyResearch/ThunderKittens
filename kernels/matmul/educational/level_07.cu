
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
#define NUM_WORKERS  (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A, B, C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>(); 
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>(); 

    int tic = 0;
    int toc = 1;
    
    rt_fl<16,BLOCK_SIZE> C_accum;
    rt_fl<16,BLOCK_SIZE> C_accum_cpy;

    int row = blockIdx.y; 
    int col = blockIdx.x; 

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid/4;

    int condition = (threadIdx.x == 0 && threadIdx.y == 0 & blockIdx.x == 0);

    __shared__ semaphore bar; 
    if (threadIdx.x == 0) { // this should be on thread and not warp (SA: note)
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(
            bar, 
            size_bytes<typeof(As[0])> +
            size_bytes<typeof(Bs[0])>
        );
        tma::load_async(As[tic], g.A, {0, 0, row, 0}, bar);
        tma::load_async(Bs[tic], g.B, {0, 0, 0, col}, bar);
    }
    __syncthreads();

    kittens::warp::zero(C_accum);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic^=1, toc^=1) {

        // arrive memory 
        wait(bar, tic);
        __syncthreads();

        // load next 
        if(warpgroupid == 0) {
            warpgroup::decrease_registers<32>();  
            if (threadIdx.x == 0 && tile+1 < num_tiles) { 
                tma::expect_bytes(bar, 
                    size_bytes<typeof(As[0])> + 
                    size_bytes<typeof(Bs[0])>
                );
                tma::load_async(As[toc], g.A, {0, 0, row, tile+1}, bar);
                tma::load_async(Bs[toc], g.B, {0, 0, tile+1, col}, bar);
            }
        } else {
            warpgroup::increase_registers<256>();
            warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
            warpgroup::mma_async_wait();
        }
        __syncthreads();
    }
    if ( warpgroupid == 1 ) { 
        warpgroup::store(g.C, C_accum, {0, 0, row, col});
    }
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
