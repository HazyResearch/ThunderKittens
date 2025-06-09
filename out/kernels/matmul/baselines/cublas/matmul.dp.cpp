#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <dpct/blas_utils.hpp>

#include <sycl/ext/intel/math.hpp>

#include <dpct/lib_common_utils.hpp>

// Error checking macro
/*
DPCT1001:612: The statement could not be removed.
*/
/*
DPCT1000:613: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:614: SYCL reports errors using exceptions and does not use error codes.
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

// Function to initialize a matrix with random values
void initMatrix(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to convert float to __nv_bfloat16
void convertFloatToBF16(const std::vector<float> &src,
                        std::vector<sycl::ext::oneapi::bfloat16> &dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = sycl::ext::intel::math::float2bfloat16(src[i]);
    }
}

// Function to perform mixed-precision matrix multiplication using cuBLAS
void matrixMultiplyMixedPrecision(dpct::blas::descriptor_ptr handle,
                                  const sycl::ext::oneapi::bfloat16 *A,
                                  const sycl::ext::oneapi::bfloat16 *B,
                                  float *C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    dpct::blas::gemm(handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::nontrans, m, n, k, &alpha, A,
                     dpct::library_data_t::real_bfloat16, m, B,
                     dpct::library_data_t::real_bfloat16, k, &beta, C,
                     dpct::library_data_t::real_float, m,
                     dpct::library_data_t::real_float);
}

// Benchmark function
void benchmark(int m, int n, int k) try {
    dpct::blas::descriptor_ptr handle;
    handle = new dpct::blas::descriptor();

    // Allocate host memory
    std::vector<float> h_A_float(m * k);
    std::vector<float> h_B_float(k * n);
    std::vector<float> h_C(m * n);
    std::vector<sycl::ext::oneapi::bfloat16> h_A_bf16(m * k);
    std::vector<sycl::ext::oneapi::bfloat16> h_B_bf16(k * n);

    // Initialize matrices
    initMatrix(h_A_float, m, k);
    initMatrix(h_B_float, k, n);

    // Convert to BF16
    convertFloatToBF16(h_A_float, h_A_bf16);
    convertFloatToBF16(h_B_float, h_B_bf16);

    // Allocate device memory
    sycl::ext::oneapi::bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(
        DPCT_CHECK_ERROR(d_A = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             m * k, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(d_B = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(
                             k * n, dpct::get_in_order_queue())));
    CHECK_CUDA(DPCT_CHECK_ERROR(
        d_C = sycl::malloc_device<float>(m * n, dpct::get_in_order_queue())));

    // Copy data to device
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_A, h_A_bf16.data(),
                    m * k * sizeof(sycl::ext::oneapi::bfloat16))
            .wait()));
    CHECK_CUDA(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_B, h_B_bf16.data(),
                    k * n * sizeof(sycl::ext::oneapi::bfloat16))
            .wait()));

    // Warm-up run
    for (int i = 0; i < 10; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
    }
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    // Benchmark
    const int NUM_ITERATIONS = 10;
    dpct::event_ptr start, stop;
    start = new sycl::event();
    stop = new sycl::event();

    dpct::sync_barrier(start);
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        matrixMultiplyMixedPrecision(handle, d_A, d_B, d_C, m, n, k);
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

    // Calculate GFLOPS
    double gflops = (2.0 * m * n * k) / (avg_time * 1e9);

    std::cout << "Matrix size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl << std::endl;

    // Clean up
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())));
    CHECK_CUDA(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue())));
    delete (handle);
    dpct::destroy_event(start);
    dpct::destroy_event(stop);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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