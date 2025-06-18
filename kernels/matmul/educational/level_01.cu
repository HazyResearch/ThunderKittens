

#include <cuda_runtime.h> 
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <chrono>

using my_dtype = float;

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


void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    // Copy to device (float)
    cudaMemcpy(d_A, h_A, M*K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*4, cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

    // Launch kernel
    for(int i = 0; i < 2; i++) { // warmup
        matmul(d_A, d_B, d_C, M);
    }
    // Start timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 10;
    for(int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C, M);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M*N*4, cudaMemcpyDeviceToHost);
    std::cout << "Copied result back to host" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        // if(error > 1.0) { 
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        // }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int N;
    N = 4096;
    run_benchmark(N, N, N);
    return 0;
}


