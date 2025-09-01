#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;  // Number of consumer warp groups
constexpr int N_BLOCK = 4;  // Number of output tiles per row

#define NUM_PRODUCER_WORKERS  (4)
#define NUM_CONSUMER_WORKERS  (M_BLOCK*4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS+NUM_CONSUMER_WORKERS)*kittens::WARP_THREADS)

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A, B, C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2, N_BLOCK>();
    
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&C_tiles)[M_BLOCK][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, M_BLOCK, N_BLOCK>();

    int tic = 0;
    int toc = 1;
    
    // Accumulator for each consumer warp group
    using wide_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE*N_BLOCK>;
    rt_fl<16, BLOCK_SIZE*N_BLOCK> C_accum;

    int row = blockIdx.y * M_BLOCK; 
    int col = blockIdx.x * N_BLOCK; 

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid/4;

    // Determine type of warp group
    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0 && warpgroupid <= M_BLOCK);
    
    // Consumer index (0-based) for consumer warp groups
    int consumer_idx = is_consumer ? (warpgroupid - 1) : 0;

    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(
            bar, 
            M_BLOCK * size_bytes<typeof(As[0][0])> +
            N_BLOCK * size_bytes<typeof(Bs[0][0])>
        );
        
        // Load initial A tiles (one row per consumer)
        for (int m = 0; m < M_BLOCK; m++) {
            tma::load_async(As[tic][m], g.A, {0, 0, row + m, 0}, bar);
        }
        
        // Load initial B tiles (all columns for this thread block)
        for (int n = 0; n < N_BLOCK; n++) {
            tma::load_async(Bs[tic][n], g.B, {0, 0, 0, col + n}, bar);
        }
    }
    __syncthreads();

    if (is_consumer) {
        kittens::warp::zero(C_accum);
    }

    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic^=1, toc^=1) {

        wait(bar, tic);
        __syncthreads();

        if (is_producer) {
            warpgroup::decrease_registers<40>();
            if (threadIdx.x == 0 && tile+1 < num_tiles) {
                tma::expect_bytes(bar, 
                    M_BLOCK * size_bytes<typeof(As[0][0])> +
                    N_BLOCK * size_bytes<typeof(Bs[0][0])>
                );
                for (int m = 0; m < M_BLOCK; m++) {
                    tma::load_async(As[toc][m], g.A, {0, 0, row + m, tile+1}, bar);
                }                
                for (int n = 0; n < N_BLOCK; n++) {
                    tma::load_async(Bs[toc][n], g.B, {0, 0, tile+1, col + n}, bar);
                }
            }
        }
        else if (is_consumer) {
            warpgroup::increase_registers<232>();
            
            // Each consumer processes its assigned row of A against all columns of B
            warpgroup::mma_AB(
                C_accum,
                As[tic][consumer_idx],                // Get this consumer's A tile
                reinterpret_cast<wide_tile&>(Bs[tic][0])  // Get all B tiles as a wide tile
            );
            warpgroup::mma_async_wait();
        }
        __syncthreads();
    }

    // Store 
    if (is_consumer) {
        
        // First store the wide result to temporary tiles
        wide_tile& wide_C_temp = reinterpret_cast<wide_tile&>(C_tiles[consumer_idx][0]);
        warpgroup::store(wide_C_temp, C_accum);        
        warpgroup::sync(warpgroupid+4);
        
        // Only first warp in each consumer group stores to global memory
        if (warpid % 4 == 0) {
            for (int n = 0; n < N_BLOCK; n++) {
                tma::store_async(g.C, C_tiles[consumer_idx][n], {0, 0, row + consumer_idx, col + n});
                tma::store_async_read_wait();
            }
        }
    }
}

// Launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 
    
    // Global pointers
    using tile_gl = matmul_globals::tile_gl;
    tile_gl a_arg{A, nullptr, nullptr, N, N};
    tile_gl b_arg{B, nullptr, nullptr, N, N};
    tile_gl c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // Launch 
    int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE*M_BLOCK;
    int NEW_COL_BLOCK_SIZE = BLOCK_SIZE*N_BLOCK;
    dim3 blocks(
        (N + NEW_COL_BLOCK_SIZE - 1) / (NEW_COL_BLOCK_SIZE),
        (N + NEW_ROW_BLOCK_SIZE - 1) / (NEW_ROW_BLOCK_SIZE)
    );
    
    unsigned long mem_size = 200000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"