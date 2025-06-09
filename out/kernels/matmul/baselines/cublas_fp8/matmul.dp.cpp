#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <dpct/blas_gemm_utils.hpp>

#include <iomanip>
#include <dpct/blas_utils.hpp>

#include <cmath>

#include <dpct/lib_common_utils.hpp>

// Error checking macros
/*
DPCT1001:512: The statement could not be removed.
*/
/*
DPCT1000:513: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:514: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        dpct::err0 err = call;                                                 \
        if (err != 0) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__  \
                      << ": " << dpct::get_error_string_dummy(err)             \
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        int status = call;                                                     \
        if (status != 0) {                                                     \
            std::cerr << "cuBLAS error in " << __FILE__ << " line "            \
                      << __LINE__ << ": "                                      \
                      << "code " << status << std::endl;                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void check(dpct::err0 error) {
    /*
    DPCT1000:510: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error != 0) {
        /*
        DPCT1009:511: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:509: The statement could not be removed.
        */
        std::cerr << "CUDA error: " << dpct::get_error_string_dummy(error)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(int status) {
    if (status != 0) {
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

void benchmark(int m, int n, int k) try {
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
    sycl::ext::oneapi::bfloat16 *d_C;
    CHECK_CUDA(DPCT_CHECK_ERROR(d_A = sycl::malloc_device<__nv_fp8_e4m3>(
                                    m * k, dpct::get_in_order_queue())));
    CHECK_CUDA(DPCT_CHECK_ERROR(d_B = sycl::malloc_device<__nv_fp8_e4m3>(
                                    n * k, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(d_C = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             m * n, dpct::get_in_order_queue())));
    CHECK_CUDA(DPCT_CHECK_ERROR(d_D = sycl::malloc_device<__nv_fp8_e4m3>(
                                    m * n, dpct::get_in_order_queue())));

    // Copy to device with same layout
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_A, h_A_fp8.data(), m * k * sizeof(__nv_fp8_e4m3))
            .wait()));
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_B, h_B_fp8.data(), n * k * sizeof(__nv_fp8_e4m3))
            .wait()));

    // Create cuBLAS handle
    dpct::blas_gemm::experimental::descriptor_ptr handle;
    CHECK_CUBLAS(DPCT_CHECK_ERROR(
        handle = new dpct::blas_gemm::experimental::descriptor()));

    // Create matrix descriptors
    dpct::blas_gemm::experimental::matrix_layout_ptr matA, matB, matC, matD;
   CHECK_CUBLAS(DPCT_CHECK_ERROR(
       matA = new dpct::blas_gemm::experimental::matrix_layout_t(
           dpct::library_data_t::real_f8_e4m3, k, m, k))); // A[K,M]
    CHECK_CUBLAS(DPCT_CHECK_ERROR(
        matB = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_f8_e4m3, k, n, k))); // B[K,N]
    CHECK_CUBLAS(DPCT_CHECK_ERROR(
        matC = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_bfloat16, m, n, m))); // C[M,N] in BF16
    CHECK_CUBLAS(DPCT_CHECK_ERROR(
        matD = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_f8_e4m3, m, n, m))); // D[M,N] in FP8

    // Create operation descriptor
    dpct::blas_gemm::experimental::matmul_desc_ptr operationDesc;
    CHECK_CUBLAS(DPCT_CHECK_ERROR(
        operationDesc = new dpct::blas_gemm::experimental::matmul_desc_t(
            dpct::compute_type::f32, dpct::library_data_t::real_float)));

    // Set operation attributes - "TN" format required for FP8
    const int32_t transa = oneapi::mkl::transpose::trans;
    const int32_t transb = oneapi::mkl::transpose::nontrans;
    CHECK_CUBLAS(DPCT_CHECK_ERROR(operationDesc->set_attribute(
        dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_a,
        &transa)));
    CHECK_CUBLAS(DPCT_CHECK_ERROR(operationDesc->set_attribute(
        dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b,
        &transb)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate workspace
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB workspace
    void* workspace = nullptr;
    CHECK_CUDA(
        DPCT_CHECK_ERROR(workspace = (void *)sycl::malloc_device(
                             workspaceSize, dpct::get_in_order_queue())));

    // Query the best algorithm
    // Create preference descriptor
    int preference;
    /*
    DPCT1027:515: The call to cublasLtMatmulPreferenceCreate was replaced with 0
    because this functionality is redundant in SYCL.
    */
    checkCublas(0);
    /*
    DPCT1027:516: The call to cublasLtMatmulPreferenceSetAttribute was replaced
    with 0 because this functionality is redundant in SYCL.
    */
    checkCublas(0);
    int returnedResults = 0;
    int heuristicResult;
    checkCublas(DPCT_CHECK_ERROR(returnedResults = 1));
    std::cout << "Returned results: " << returnedResults << std::endl;

    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        CHECK_CUBLAS(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
            handle, operationDesc, &alpha, d_A, matA, d_B, matB, &beta, d_C,
            matC, d_D, matD, 0)));
    }

    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    // Benchmark runs
    const int NUM_ITERATIONS = 100;
    dpct::event_ptr start, stop;
    CHECK_CUDA(DPCT_CHECK_ERROR(start = new sycl::event()));
    CHECK_CUDA(DPCT_CHECK_ERROR(stop = new sycl::event()));

    /*
    DPCT1024:517: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CHECK_CUDA(DPCT_CHECK_ERROR(dpct::sync_barrier(start)));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CHECK_CUBLAS(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
            handle, operationDesc, &alpha, d_A, matA, d_B, matB, &beta, d_C,
            matC, d_D, matD, 0)));
    }
    /*
    DPCT1024:518: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CHECK_CUDA(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
    CHECK_CUDA(DPCT_CHECK_ERROR(stop->wait_and_throw()));

    float milliseconds = 0;
    CHECK_CUDA(DPCT_CHECK_ERROR(
        milliseconds = (stop->get_profiling_info<
                            sycl::info::event_profiling::command_end>() -
                        start->get_profiling_info<
                            sycl::info::event_profiling::command_start>()) /
                       1000000.0f));
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
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(h_D_fp8.data(), d_D, m * n * sizeof(__nv_fp8_e4m3))
            .wait()));

    // Convert FP8 to float for comparison
    for (int i = 0; i < m * n; i++) {
        h_D[i] = float(h_D_fp8[i]);  // Convert FP8 to float
    }

    // Now compare the float values
    check_result(h_D.data(), h_D_ref.data(), m, n);

    // Cleanup
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::dpct_free(workspace, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_D, dpct::get_in_order_queue())));
    CHECK_CUDA(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
    CHECK_CUDA(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
    delete (matA);
    delete (matB);
    delete (matC);
    delete (matD);
    delete (operationDesc);
    delete (handle);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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