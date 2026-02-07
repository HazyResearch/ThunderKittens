#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <chrono>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;
__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
   __shared__ __nv_bfloat16 As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE]; // Shared memory tiles for A and B
   int tx = threadIdx.x, bx = blockIdx.x; 
   int by = blockIdx.y, ty = threadIdx.y; 
   int row = by * BLOCK_SIZE + ty; // Thread indices
   int col = bx * BLOCK_SIZE + tx;
   float sum = 0.0f; // Accumulator for dot product
   for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
       As[ty][tx] = A[row * N + tile * BLOCK_SIZE + tx]; // Load data into shared memory
       Bs[ty][tx] = B[col * N + tile * BLOCK_SIZE + ty]; // B stored transposed: B^T[col][k]
       __syncthreads(); // Synchronize to ensure all threads have loaded the data
       #pragma unroll // Compute partial dot product in registers
       for (int k = 0; k < BLOCK_SIZE; ++k) {
           sum += __bfloat162float(As[ty][k] * Bs[k][tx]);
       }
       __syncthreads(); // Synchronize before loading next tile
   }
   C[row * N + col] = __float2bfloat16(sum); // Write result to global memory
}

// launch kernel
void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N); // kernel assumes M=N=K
}

#include "launch.cu"
