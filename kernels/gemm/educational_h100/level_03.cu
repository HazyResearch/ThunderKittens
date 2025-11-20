#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>

using my_dtype = __nv_bfloat16; 

constexpr int BLOCK_SIZE = 32;
__global__ void kernel(my_dtype* A, my_dtype* B, my_dtype* C, int N) {
   __shared__ my_dtype As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE]; // Shared memory tiles for A and B
   int tx = threadIdx.x, bx = blockIdx.x; 
   int by = blockIdx.y, ty = threadIdx.y; 
   int row = by * BLOCK_SIZE + ty; // Thread indices
   int col = bx * BLOCK_SIZE + tx;
   my_dtype sum = 0.0f; // Accumulator for dot product
   for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
       As[ty][tx] = A[row * N + tile * BLOCK_SIZE + tx]; // Load data into shared memory
       Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
       __syncthreads(); // Synchronize to ensure all threads have loaded the data
       #pragma unroll // Compute partial dot product in registers
       for (int k = 0; k < BLOCK_SIZE; ++k) {
           sum += As[ty][k] * Bs[k][tx];
       }
       __syncthreads(); // Synchronize before loading next tile
   }
   C[row * N + col] = sum; // Write result to global memory
}

// launch kernel
void matmul(my_dtype* A, my_dtype* B, my_dtype* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch.cu"
