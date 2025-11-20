#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>


#include <cuda_runtime.h>

using my_dtype = __nv_bfloat16; 

__global__ void kernel(my_dtype* A, my_dtype* B, my_dtype* C, int N) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
  
   if (row < N && col < N) {
       my_dtype sum = 0.0f;
       for (int k = 0; k < N; k++) {
           sum += A[row * N + k] * B[k * N + col];
       }
       C[row * N + col] = sum;
   }
}

// launch kernel
int BLOCK_SIZE = 32;
void matmul(my_dtype* A, my_dtype* B, my_dtype* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch.cu"
