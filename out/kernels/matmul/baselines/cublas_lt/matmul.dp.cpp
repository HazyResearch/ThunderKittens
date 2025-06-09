#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_gemm_utils.hpp>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <dpct/blas_utils.hpp>

#include <cmath>

#include <sycl/ext/intel/math.hpp>

#include <dpct/lib_common_utils.hpp>

void check(dpct::err0 error) {
    /*
    DPCT1000:597: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error != 0) {
        /*
        DPCT1009:598: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:596: The statement could not be removed.
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
    std::vector<sycl::ext::oneapi::bfloat16> h_A(m * k);
    std::vector<sycl::ext::oneapi::bfloat16> h_B(k * n);
    for(int i = 0; i < m * k; i++) {
        h_A[i] = sycl::ext::intel::math::float2bfloat16(dist(gen));
    }
    for(int i = 0; i < k * n; i++) {
        h_B[i] = sycl::ext::intel::math::float2bfloat16(dist(gen));
    }
    std::vector<sycl::ext::oneapi::bfloat16> h_C(
        m * n, sycl::ext::oneapi::bfloat16(0.0f));

    // Allocate device memory
    sycl::ext::oneapi::bfloat16 *d_A, *d_B, *d_C;
    uint8_t *workspace;
    check(DPCT_CHECK_ERROR(workspace = (uint8_t *)sycl::malloc_device(
                               32 * 1024 * 1024, dpct::get_in_order_queue())));
    check(
        DPCT_CHECK_ERROR(d_A = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             m * k, dpct::get_in_order_queue())));
    check(
        DPCT_CHECK_ERROR(d_B = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             k * n, dpct::get_in_order_queue())));
    check(
        DPCT_CHECK_ERROR(d_C = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             m * n, dpct::get_in_order_queue())));

    // Copy data to device
    check(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_A, h_A.data(),
                    m * k * sizeof(sycl::ext::oneapi::bfloat16))
            .wait()));
    check(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_B, h_B.data(),
                    k * n * sizeof(sycl::ext::oneapi::bfloat16))
            .wait()));

    // Initialize cuBLASLt
    dpct::blas_gemm::experimental::descriptor_ptr handle;
    checkCublas(DPCT_CHECK_ERROR(
        handle = new dpct::blas_gemm::experimental::descriptor()));

    // Configure matrix descriptors
    dpct::blas_gemm::experimental::matrix_layout_ptr matA, matB,
        matC; // apparently column major is strongly preferred (???)
    checkCublas(DPCT_CHECK_ERROR(
        matA = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_bfloat16, m, k, m)));
    checkCublas(DPCT_CHECK_ERROR(
        matB = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_bfloat16, k, n, k)));
    checkCublas(DPCT_CHECK_ERROR(
        matC = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_bfloat16, m, n, m)));

    // Configure matrix multiplication descriptor
    dpct::blas_gemm::experimental::matmul_desc_ptr matmulDesc;
    checkCublas(DPCT_CHECK_ERROR(
        matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(
            dpct::compute_type::f32_fast_bf16,
            dpct::library_data_t::real_float)));

    // Set matrix operation parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t workspaceSize = 32 * 1024 * 1024;
    
    // Create preference descriptor
    int preference;
    /*
    DPCT1027:600: The call to cublasLtMatmulPreferenceCreate was replaced with 0
    because this functionality is redundant in SYCL.
    */
    checkCublas(0);
    /*
    DPCT1027:601: The call to cublasLtMatmulPreferenceSetAttribute was replaced
    with 0 because this functionality is redundant in SYCL.
    */
    checkCublas(0);

    // Query the best algorithm
    int returnedResults = 0;
    int heuristicResult;
    checkCublas(DPCT_CHECK_ERROR(returnedResults = 1));
    std::cout << "Returned results: " << returnedResults << std::endl;

    // Warmup iterations
    for (int i = 0; i < 10; i++) {
        checkCublas(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
            handle, matmulDesc, &alpha, d_A, matA, d_B, matB, &beta, d_C, matC,
            d_C, matC, 0)));
    }
    
    // Synchronize before timing
    check(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    // Timing iterations
    const int NUM_ITERATIONS = 10;
    dpct::event_ptr start, stop;
    start = new sycl::event();
    stop = new sycl::event();

    dpct::sync_barrier(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        checkCublas(DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
            handle, matmulDesc, &alpha, d_A, matA, d_B, matB, &beta, d_C, matC,
            d_C, matC, 0)));
    }
    dpct::sync_barrier(stop);
    stop->wait_and_throw();

    float milliseconds = 0;
    milliseconds =
        (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
         start->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    float avg_time = milliseconds / NUM_ITERATIONS;
    
    // Calculate TFLOPS
    double flops = 2.0 * m * n * k; // multiply-add counts as 2 operations
    double tflops = (flops / (avg_time * 1e-3)) / 1e12;
    
    std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time 
              << " ms, Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    
    // Verify correctness on random indices
    // Copy matrices back to host
    check(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(h_C.data(), d_C,
                    m * n * sizeof(sycl::ext::oneapi::bfloat16))
            .wait()));

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
            expected +=
                sycl::ext::intel::math::bfloat162float(h_A[j * m + row]) *
                sycl::ext::intel::math::bfloat162float(h_B[col * k + j]);
        }
        expected =
            alpha * expected +
            beta * sycl::ext::intel::math::bfloat162float(h_C[row * n + col]);

        // Get actual value and convert to float for comparison
        float actual =
            sycl::ext::intel::math::bfloat162float(h_C[col * m + row]);

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
    /*
    DPCT1026:599: The call to cublasLtMatmulPreferenceDestroy was removed
    because this functionality is redundant in SYCL.
    */
    delete (matA);
    delete (matB);
    delete (matC);
    delete (matmulDesc);
    delete (handle);

    dpct::dpct_free(d_A, dpct::get_in_order_queue());
    dpct::dpct_free(d_B, dpct::get_in_order_queue());
    dpct::dpct_free(d_C, dpct::get_in_order_queue());

    return tflops;
}

int main() {
    // Initialize CUDA
    int device = 0;
    /*
    DPCT1093:602: The "device" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(device);

    dpct::device_info deviceProp;
    dpct::get_device(device).get_device_info(deviceProp);
    std::cout << "Running on GPU: " << deviceProp.get_name() << std::endl;

    // Matrix sizes to benchmark
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    
    // Run benchmarks
    for (int size : sizes) {
        benchmark_matmul(size);
    }
    
    return 0;
}