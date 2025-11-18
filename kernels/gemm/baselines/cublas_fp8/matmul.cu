#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <iomanip>

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << "code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // sum += a[i * K + k] * b[j * K + k]; // mma_ABt
                sum += a[i * K + k] * b[k * N + j]; // mma_AB
            }
            c[i * N + j] = sum;
        }
    }
}

void check_result(float* h_C, float* h_C_ref, int M, int N) {
    float max_error = 0.0f;
    int error_count = 0;
    
    // Same tolerance and error reporting as your code
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(1) { // large tolerance because of fp8 vs fp32 numerics
            if(error_count < 25) {
                std::cout << "Error at row " << i / N << " col " << i % N 
                         << ": " << h_C[i] << " != " << h_C_ref[i] 
                         << " (ref)" << std::endl;
            }
            else if(error_count == 25) {
                std::cout << "Too many errors to show them all.\n";
            }
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
}

void benchmark(int m, int n, int k) {
    // Align dimensions
    m = (m + 15) & ~15;
    n = (n + 15) & ~15;
    k = (k + 15) & ~15;

    // Initialize host memory with same layout as your code
    std::vector<float> h_A(m * k);  // A[M,K]
    std::vector<float> h_B(n * k);  // B[N,K]
    std::vector<float> h_D(m * n);
    std::vector<float> h_D_ref(m * n);

    // Initialize with random values just like your code
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (int i = 0; i < m * k; ++i) h_A[i] = 1; //dis(gen) * 0.5f;
    for (int i = 0; i < n * k; ++i) h_B[i] = dis(gen) * 0.5f;

    // Convert to FP8
    std::vector<__nv_fp8_e4m3> h_A_fp8(m * k);
    std::vector<__nv_fp8_e4m3> h_B_fp8(n * k);
    for (int i = 0; i < m * k; ++i) h_A_fp8[i] = __nv_fp8_e4m3(h_A[i]);
    for (int i = 0; i < n * k; ++i) h_B_fp8[i] = __nv_fp8_e4m3(h_B[i]);

    // Allocate device memory
    __nv_fp8_e4m3 *d_A, *d_B, *d_D;
    __nv_bfloat16 *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_B, n * k * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_D, m * n * sizeof(__nv_fp8_e4m3)));

    // Copy to device with same layout
    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp8.data(), m * k * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_fp8.data(), n * k * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matrix descriptors
    cublasLtMatrixLayout_t matA, matB, matC, matD;
   CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA, CUDA_R_8F_E4M3, k, m, k));  // A[K,M]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB, CUDA_R_8F_E4M3, k, n, k));  // B[K,N]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC, CUDA_R_16BF, m, n, m));     // C[M,N] in BF16
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matD, CUDA_R_8F_E4M3, m, n, m));  // D[M,N] in FP8


    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // Set operation attributes - "TN" format required for FP8
    const int32_t transa = CUBLAS_OP_T;
    const int32_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate workspace
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB workspace
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Query the best algorithm
    // Create preference descriptor
    cublasLtMatmulPreference_t preference;
    checkCublas(cublasLtMatmulPreferenceCreate(&preference));
    checkCublas(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    checkCublas(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, matA, matB, matC, matD, preference, 1, &heuristicResult, &returnedResults
    ));
    std::cout << "Returned results: " << returnedResults << std::endl;

    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_D, matD,
            &heuristicResult.algo,       
            workspace,
            workspaceSize,
            0                   // Default stream
        ));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    const int NUM_ITERATIONS = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_D, matD,
            &heuristicResult.algo,
            workspace,
            workspaceSize,
            0
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / NUM_ITERATIONS;

    // Calculate TFLOPS
    double num_ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k); // multiply-add counts as 2
    double seconds = avg_time / 1000.0; // convert ms to seconds
    double tflops = (num_ops / seconds) / 1e12;

    std::cout << "Matrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl << std::endl;

    // Get cuBLAS result
    cpu_gemm(h_A.data(), h_B.data(), h_D_ref.data(), m, n, k);

    // Allocate FP8 host buffer
    std::vector<__nv_fp8_e4m3> h_D_fp8(m * n);
    CHECK_CUDA(cudaMemcpy(h_D_fp8.data(), d_D, m * n * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));

    // Convert FP8 to float for comparison
    for (int i = 0; i < m * n; i++) {
        h_D[i] = float(h_D_fp8[i]);  // Convert FP8 to float
    }

    // Now compare the float values
    check_result(h_D.data(), h_D_ref.data(), m, n);

    // Cleanup
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasLtMatrixLayoutDestroy(matA);
    cublasLtMatrixLayoutDestroy(matB);
    cublasLtMatrixLayoutDestroy(matC);
    cublasLtMatrixLayoutDestroy(matD);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(handle);
}

int main() {
    // Benchmark different matrix sizes
    // std::vector<int> sizes = {3072, 4096, 6144, 8192, 12288, 16384};
    std::vector<int> sizes = {2048};
    
    for (int size : sizes) {
        benchmark(size, size, size);
    }

    return 0;
}