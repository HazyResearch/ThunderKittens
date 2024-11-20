#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

double benchmark_matmul(int matrix_size) {
    std::cout << "\nBenchmarking size: " << matrix_size << "x" << matrix_size << std::endl;
    
    // Initialize dimensions
    const int m = matrix_size;
    const int n = matrix_size;
    const int k = matrix_size;
    
    // Allocate host memory
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(std::sqrt(1.0f/k)));
    std::vector<__nv_bfloat16> h_A(m * k);
    std::vector<__nv_bfloat16> h_B(k * n);
    for(int i = 0; i < m * k; i++) {
        h_A[i] = __float2bfloat16(dist(gen));
    }
    for(int i = 0; i < k * n; i++) {
        h_B[i] = __float2bfloat16(dist(gen));
    }
    std::vector<__nv_bfloat16> h_C(m * n, __nv_bfloat16(0.0f));
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    uint8_t *workspace;
    check(cudaMalloc(&workspace, 32 * 1024 * 1024));
    check(cudaMalloc(&d_A, m * k * sizeof(__nv_bfloat16)));
    check(cudaMalloc(&d_B, k * n * sizeof(__nv_bfloat16)));
    check(cudaMalloc(&d_C, m * n * sizeof(__nv_bfloat16)));
    
    // Copy data to device
    check(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    
    // Initialize cuBLASLt
    cublasLtHandle_t handle;
    checkCublas(cublasLtCreate(&handle));
    
    // Configure matrix descriptors
    cublasLtMatrixLayout_t matA, matB, matC; // apparently column major is strongly preferred (???)
    checkCublas(cublasLtMatrixLayoutCreate(&matA, CUDA_R_16BF, m, k, m));
    checkCublas(cublasLtMatrixLayoutCreate(&matB, CUDA_R_16BF, k, n, k));
    checkCublas(cublasLtMatrixLayoutCreate(&matC, CUDA_R_16BF, m, n, m));
    
    // Configure matrix multiplication descriptor
    cublasLtMatmulDesc_t matmulDesc;
    checkCublas(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F));
    
    // Set matrix operation parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t workspaceSize = 32 * 1024 * 1024;
    
    // Create preference descriptor
    cublasLtMatmulPreference_t preference;
    checkCublas(cublasLtMatmulPreferenceCreate(&preference));
    checkCublas(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    // Query the best algorithm
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    checkCublas(cublasLtMatmulAlgoGetHeuristic(
        handle, matmulDesc, matA, matB, matC, matC, preference, 1, &heuristicResult, &returnedResults
    ));
    std::cout << "Returned results: " << returnedResults << std::endl;

    // Warmup iterations
    for (int i = 0; i < 10; i++) {
        checkCublas(cublasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matC,
            &heuristicResult.algo,
            workspace, workspaceSize,
            0
        ));
    }
    
    // Synchronize before timing
    check(cudaDeviceSynchronize());
    
    // Timing iterations
    const int NUM_ITERATIONS = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        checkCublas(cublasLtMatmul(
            handle,
            matmulDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matC,
            &heuristicResult.algo,
            workspace, workspaceSize,
            0
        ));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / NUM_ITERATIONS;
    
    // Calculate TFLOPS
    double flops = 2.0 * m * n * k; // multiply-add counts as 2 operations
    double tflops = (flops / (avg_time * 1e-3)) / 1e12;
    
    std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time 
              << " ms, Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    
    // Verify correctness on random indices
    // Copy matrices back to host
    check(cudaMemcpy(h_C.data(), d_C, m * n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Seed random number generator
    std::uniform_int_distribution<> dis_m(0, m-1);
    std::uniform_int_distribution<> dis_n(0, n-1);

    // Check 50 random positions
    for (int i = 0; i < 50; i++) {
        int row = dis_m(gen);
        int col = dis_n(gen);
        
        // Calculate expected value
        float expected = 0.0f;  // Use float for intermediate computation
        for (int j = 0; j < k; j++) {
            expected += __bfloat162float(h_A[j * m + row]) * __bfloat162float(h_B[col * k + j]);
        }
        expected = alpha * expected + beta * __bfloat162float(h_C[row * n + col]);
        
        // Get actual value and convert to float for comparison
        float actual = __bfloat162float(h_C[col * m + row]);
        
        // Compare with larger tolerance due to bf16 precision
        float rel_error = std::abs(actual - expected) / std::abs(expected);
        if (rel_error > 0.01) {  // Increased tolerance for bf16
            std::cout << "Verification failed at position [" << row << "," << col << "]" << std::endl;
            std::cout << "Expected: " << expected << ", Got: " << actual << ", Relative Error: " << rel_error << std::endl;
        }
        else {
            if(i < 5) {
                std::cout << "Verification passed at position [" << row << "," << col << "]" << " with values " << expected << " and " << actual << std::endl;
            }
        }
    }

    // Cleanup
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(matA);
    cublasLtMatrixLayoutDestroy(matB);
    cublasLtMatrixLayoutDestroy(matC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return tflops;
}

int main() {
    // Initialize CUDA
    int device = 0;
    cudaSetDevice(device);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "Running on GPU: " << deviceProp.name << std::endl;
    
    // Matrix sizes to benchmark
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    
    // Run benchmarks
    for (int size : sizes) {
        benchmark_matmul(size);
    }
    
    return 0;
}