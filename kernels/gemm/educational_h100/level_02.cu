#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <chrono>


#include <cuda_runtime.h>

__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
  
   if (row < N && col < N) {
       float sum = 0.0f;
       for (int k = 0; k < N; k++) {
           sum += __bfloat162float(A[row * N + k] * B[k * N + col]);
       }
       C[row * N + col] = __float2bfloat16(sum);
   }
}

// launch kernel
int BLOCK_SIZE = 32;
void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch.cu"
