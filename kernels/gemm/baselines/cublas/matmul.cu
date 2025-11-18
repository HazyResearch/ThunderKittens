#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to initialize a matrix with random values
void initMatrix(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to convert float to __nv_bfloat16
void convertFloatToBF16(const std::vector<float>& src, std::vector<__nv_bfloat16>& dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Function to perform mixed-precision matrix multiplication using cuBLAS
void matrixMultiplyMixedPrecision(cublasHandle_t handle, const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                 &alpha, A, CUDA_R_16BF, m,
                 B, CUDA_R_16BF, k,
                 &beta, C, CUDA_R_32F, m,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Benchmark function
void benchmark(int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate host memory
    std::vector<float> h_A_float(m * k);
    std::vector<float> h_B_float(k * n);
    std::vector<float> h_C(m * n);
    std::vector<__nv_bfloat16> h_A_bf16(m * k);
    std::vector<__nv_bfloat16> h_B_bf16(k * n);

    // Initialize matrices
    initMatrix(h_A_float, m, k);
    initMatrix(h_B_float, k, n);

    // Convert to BF16
    convertFloatToBF16(h_A_float, h_A_bf16);
    convertFloatToBF16(h_B_float, h_B_bf16);

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf16.data(), m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf16.data(), k * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Warm-up run
    for (int i = 0; i < 10; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    const int NUM_ITERATIONS = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / NUM_ITERATIONS;

    // Calculate GFLOPS
    double gflops = (2.0 * m * n * k) / (avg_time * 1e9);

    std::cout << "Matrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Benchmark different matrix sizes
    benchmark(1024, 1024, 1024);
    benchmark(2048, 2048, 2048);
    benchmark(4096, 4096, 4096);
    benchmark(8192, 8192, 8192);
    benchmark(16384, 16384, 16384);

    return 0;
}