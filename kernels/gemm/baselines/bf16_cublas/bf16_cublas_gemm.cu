/***************************************************************************************************
 * cuBLAS BF16 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__ << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Fixed iteration counts
static constexpr int warmup_iters = 500;
static constexpr int profiling_iters = 100;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Simple reference GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Each thread computes one output element
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void reference_gemm_kernel(
    __nv_bfloat16* D,
    __nv_bfloat16 const* A,
    __nv_bfloat16 const* B,
    int M, int N, int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a = __bfloat162float(A[row * K + k]);      // A[row, k] - RowMajor MxK
      float b = __bfloat162float(B[col * K + k]);      // B[col, k] - ColMajor NxK (N rows of K elements)
      acc += a * b;
    }
    D[row * N + col] = __float2bfloat16(acc);          // D[row, col] - RowMajor MxN
  }
}

void launch_reference_gemm(__nv_bfloat16* D, __nv_bfloat16 const* A,
                           __nv_bfloat16 const* B, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  reference_gemm_kernel<<<grid, block>>>(D, A, B, M, N, K);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization kernel - uniform distribution [-1, 1]
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init_random_kernel(__nv_bfloat16* data, size_t count, uint64_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Splitmix64 hash for uniform random bits
    uint64_t x = seed + idx;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    // Upper 24 bits to float in [0,1), then scale to [-1,1)
    float u = (float)(x >> 40) * (1.0f / 16777216.0f);  // 2^24 = 16777216
    float val = u * 2.0f - 1.0f;
    data[idx] = __float2bfloat16(val);
  }
}

void init_random(__nv_bfloat16* data, size_t count, uint64_t seed) {
  dim3 block(256);
  dim3 grid((count + 255) / 256);
  init_random_kernel<<<grid, block>>>(data, count, seed);
}

__global__ void fill_zero_kernel(__nv_bfloat16* data, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = __float2bfloat16(0.0f);
  }
}

void fill_zero(__nv_bfloat16* data, size_t count) {
  int block_size = 256;
  int grid_size = (count + block_size - 1) / block_size;
  fill_zero_kernel<<<grid_size, block_size>>>(data, count);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Correctness verification
///////////////////////////////////////////////////////////////////////////////////////////////////

void verify_gemm(__nv_bfloat16 const* d_D, __nv_bfloat16 const* d_D_ref, int M, int N) {
  size_t count = size_t(M) * N;
  std::vector<__nv_bfloat16> h_D(count);
  std::vector<__nv_bfloat16> h_D_ref(count);

  cudaMemcpy(h_D.data(), d_D, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D_ref.data(), d_D_ref, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  double abs_sum = 0.0, abs_max = 0.0;
  double err_sum = 0.0, err_max = 0.0;

  for (size_t i = 0; i < count; ++i) {
    float val = std::abs(__bfloat162float(h_D[i]));
    float ref = std::abs(__bfloat162float(h_D_ref[i]));
    float err = std::abs(val - ref);

    abs_sum += val;
    abs_max = std::max(abs_max, (double)val);
    err_sum += err;
    err_max = std::max(err_max, (double)err);
  }

  double abs_mean = abs_sum / count;
  double err_mean = err_sum / count;

  std::cout << "abs mean: " << abs_mean << ", abs max: " << abs_max
            << ", err mean: " << err_mean << ", err max: " << err_max << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

void cublas_gemm(
    cublasHandle_t handle,
    __nv_bfloat16 const* A,
    __nv_bfloat16 const* B,
    __nv_bfloat16* D,
    int M, int N, int K) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // A: RowMajor MxK (= ColMajor KxM), B: ColMajor NxK (= RowMajor NxK), D: RowMajor MxN (= ColMajor NxM)
  // D[m,n] = sum_k A[m,k] * B[n,k]
  // In col-major: D' = B'^T * A' where B' is KxN, A' is KxM, D' is NxM
  CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_T,    // B' (KxN) transposed gives NxK
      CUBLAS_OP_N,    // A' (KxM) as-is gives KxM
      N, M, K,        // output NxM (= row-major MxN)
      &alpha,
      B, CUDA_R_16BF, K,   // B: RowMajor NxK = ColMajor KxN, ld = K
      A, CUDA_R_16BF, K,   // A: RowMajor MxK = ColMajor KxM, ld = K
      &beta,
      D, CUDA_R_16BF, N,   // D: RowMajor MxN = ColMajor NxM, ld = N
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark function
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(int M, int N, int K) {
  // Cooldown between configurations
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // L2 cache eviction - multiple buffer groups
  int l2_cache_size;
  cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
  const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);  // bytes for bf16
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

  // Allocate buffer groups
  std::vector<__nv_bfloat16*> blocks_A(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_B(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_D(arg_group_count);
  __nv_bfloat16* block_D_ref;

  size_t size_A = size_t(M) * K;
  size_t size_B = size_t(K) * N;
  size_t size_D = size_t(M) * N;

  CHECK_CUDA(cudaMalloc(&block_D_ref, size_D * sizeof(__nv_bfloat16)));

  uint64_t seed = 2024;
  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(__nv_bfloat16)));

    // Initialize with uniform random [-1, 1]
    init_random(blocks_A[i], size_A, seed + i * 100);
    init_random(blocks_B[i], size_B, seed + i * 100 + 1);
    fill_zero(blocks_D[i], size_D);
  }
  fill_zero(block_D_ref, size_D);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute reference GEMM
  launch_reference_gemm(block_D_ref, blocks_A[0], blocks_B[0], M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // cuBLAS Benchmark
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    int idx = i % arg_group_count;
    cublas_gemm(handle, blocks_A[idx], blocks_B[idx], blocks_D[idx], M, N, K);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < profiling_iters; ++i) {
    int idx = i % arg_group_count;
    cublas_gemm(handle, blocks_A[idx], blocks_B[idx], blocks_D[idx], M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  double runtime_ms = static_cast<double>(milliseconds) / profiling_iters;
  double runtime_s = runtime_ms / 1000.0;
  int64_t flops = int64_t(2) * M * N * K;
  double tflops = (double(flops) / 1e12) / runtime_s;

  std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
  std::cout << "Performance: " << tflops << " TFLOP/s" << std::endl;

  // Verify correctness
  fill_zero(blocks_D[0], size_D);
  cublas_gemm(handle, blocks_A[0], blocks_B[0], blocks_D[0], M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());
  verify_gemm(blocks_D[0], block_D_ref, M, N);

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaFree(blocks_A[i]));
    CHECK_CUDA(cudaFree(blocks_B[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
  CHECK_CUDA(cudaFree(block_D_ref));

  CHECK_CUBLAS(cublasDestroy(handle));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "cuBLAS BF16 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "Accumulator: FP32, Output: BF16" << std::endl;
  std::cout << "Warmup: " << warmup_iters << ", Profiling: " << profiling_iters << std::endl;

  benchmark(1024, 1024, 1024);
  benchmark(2048, 2048, 2048);
  benchmark(4096, 4096, 4096);
  benchmark(8192, 8192, 8192);
  benchmark(16384, 16384, 16384);

  return 0;
}