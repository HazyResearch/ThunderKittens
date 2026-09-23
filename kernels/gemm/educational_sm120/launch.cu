#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../common.cuh"

int run_benchmark(size_t N) {
    cudaError_t cudaStatus;
    std::cout << "--------------------  N=" << N << "  --------------------\n";

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    std::cout << "Allocated host memory" << std::endl;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (size_t i = 0; i < N * N; ++i) h_A[i] = dis(gen);
    for (size_t i = 0; i < N * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    __nv_bfloat16 *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, N*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, N*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, N*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, N*N*sizeof(__nv_bfloat16));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[N * N];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[N * N];
    for (size_t i = 0; i < N * N; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (size_t i = 0; i < N * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    cudaMemcpy(d_A, h_A_bf16, N*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, N*N*2, cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A, d_B, N, N, N);
    cudaDeviceSynchronize();
    std::cout << "Computed reference GEMM on device" << std::endl;
    printf("\n");

    for(int i = 0; i < 2; i++) {
        matmul(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 1;
    for(int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    double flops = double(2.0) * N * N * N;
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[N * N];
    __nv_bfloat16 *h_C_ref_bf16 = new __nv_bfloat16[N * N];
    cudaMemcpy(h_C_bf16, d_C, N*N*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref_bf16, d_C_ref, N*N*2, cudaMemcpyDeviceToHost);
    std::cout << "Copied result back to host" << std::endl;

    float *h_C_ref = new float[N * N];
    for (size_t i = 0; i < N * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);
    for (size_t i = 0; i < N * N; ++i) h_C_ref[i] = __bfloat162float(h_C_ref_bf16[i]);
    std::cout << "Converted result back to float" << std::endl;

    float max_error = 0.0f;
    int error_count = 0;
    for (size_t i = 0; i < N * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if( error > 0.2 ) {
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
    std::cout << "Total count: " << int(N * N) << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    delete[] h_C_ref_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);

    return 0;
}

int main() {
    run_benchmark(4096);
    return 0;
}
