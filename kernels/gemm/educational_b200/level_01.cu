#include <cuda_runtime.h>
#include <iostream>
#include <random>
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

int BLOCK_SIZE = 32;
void matmul(my_dtype* A, my_dtype* B, my_dtype* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

void cpu_gemm(float* a, float* b, float* c, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int run_benchmark(size_t N) {
    cudaError_t cudaStatus;
    std::cout << "--------------------  N=" << N << "  --------------------\n";

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    float *h_C_ref = new float[N * N];

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (size_t i = 0; i < N * N; ++i) h_A[i] = dis(gen);
    for (size_t i = 0; i < N * N; ++i) h_B[i] = dis(gen);

    cpu_gemm(h_A, h_B, h_C_ref, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaMemcpy(d_A, h_A, N*N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*4, cudaMemcpyHostToDevice);

    for (int i = 0; i < 2; i++) matmul(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 10;
    for (int i = 0; i < ITERS; i++) matmul(d_A, d_B, d_C, N);
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

    cudaMemcpy(h_C, d_C, N*N*4, cudaMemcpyDeviceToHost);

    int error_count = 0;
    float max_error = 0.0f;
    for (size_t i = 0; i < N * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if (error > .01f) {
            if (error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if (error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

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
    run_benchmark(1024);
    return 0;
}
