
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int QSIZE = 2; // Double buffering
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
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[QSIZE] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, QSIZE>(); 
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[QSIZE] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, QSIZE>(); 
    
    int row = blockIdx.y, col = blockIdx.x; 
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int warpid        = kittens::warpid();
    const int warpgroupid   = warpid/4;
    const int num_consumers = (NUM_THREADS / 128) - 1;

    __shared__ semaphore full[QSIZE], empty[QSIZE]; 
    if (threadIdx.x == 0) { // this should be on thread and not warp (SA: note)
        for (int i = 0; i < QSIZE; ++i) {
            init_semaphore(full[i], 0, 1);
            init_semaphore(empty[i], num_consumers, 0);
        }
    }
    __syncthreads();

    if (warpgroupid == 0) { // producer
        warpgroup::decrease_registers<32>();  
        if (warpgroup::laneid() == 0) {
            int p = 0, qidx = 0;
            for (int tile = 0; tile < num_tiles; ++tile, ++qidx) {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                wait(empty[qidx], p);
                tma::expect_bytes(full[qidx], size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>);
                tma::load_async(As[qidx], g.A, {0, 0, row, tile}, full[qidx]);
                tma::load_async(Bs[qidx], g.B, {0, 0, tile, col}, full[qidx]);
            }
        }

    } else { // consumer
        warpgroup::increase_registers<256>();
        rt_fl<16,BLOCK_SIZE> C_accum;
        kittens::warp::zero(C_accum);

        if (warpgroup::laneid() == 0)
            for (int i = 0; i < QSIZE; ++i) arrive(empty[i], 1);

        int p = 0, qidx = 0;
        for (int tile = 0; tile < num_tiles; ++tile, ++qidx) {
            if (qidx == QSIZE) { qidx = 0; p ^= 1; }
            wait(full[qidx], p);
            warpgroup::mma_AB(C_accum, As[qidx], Bs[qidx]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) arrive(empty[qidx], 1);
        }

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
