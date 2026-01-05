/***************************************************************************************************
 * cuBLASLt BF16 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>

#include "../../common.cuh"

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
            std::cerr << "cuBLASLt error in " << __FILE__ << " line " << __LINE__ << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static constexpr int warmup_iters = 500;
static constexpr int profiling_iters = 100;

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLASLt GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtGemm {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;
  void* workspace;
  size_t workspaceSize;

  void init(int M, int N, int K) {
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matmul descriptor
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // D[m,n] = sum_k A[m,k] * B[n,k]
    // A: RowMajor MxK = ColMajor KxM, B: ColMajor NxK = RowMajor NxK = ColMajor KxN
    // D: RowMajor MxN = ColMajor NxM
    // In col-major: D' = B'^T * A' where B' is KxN, A' is KxM, D' is NxM
    cublasOperation_t transA = CUBLAS_OP_T;  // B' (KxN) transposed gives NxK
    cublasOperation_t transB = CUBLAS_OP_N;  // A' (KxM) as-is gives KxM
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Layout for B (cuBLAS "A"): RowMajor NxK = ColMajor KxN, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, K, N, K));
    // Layout for A (cuBLAS "B"): RowMajor MxK = ColMajor KxM, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM, ld=N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // Workspace
    workspaceSize = 32 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));

    // Get best algorithm
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
                                                 preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
      std::cerr << "No algorithm found!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void run(__nv_bfloat16 const* A, __nv_bfloat16 const* B, __nv_bfloat16* D, cudaStream_t stream = nullptr) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Note: B is first arg, A is second arg (for the transpose trick)
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, &alpha,
                                 B, layoutA,   // "A" in cublasLt = our B
                                 A, layoutB,   // "B" in cublasLt = our A
                                 &beta,
                                 D, layoutD,
                                 D, layoutD,
                                 &heuristic.algo, workspace, workspaceSize, stream));
  }

  void destroy() {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutD));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
    CHECK_CUBLAS(cublasLtDestroy(handle));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark function
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(int M, int N, int K) {
  // Cooldown between configurations
  sleep_ms(500);

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // L2 cache eviction - multiple buffer groups
  int l2_cache_size;
  cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
  const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
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

    fill<__nv_bfloat16, FillMode::RANDOM>(blocks_A[i], size_A, seed + i * 100, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::RANDOM>(blocks_B[i], size_B, seed + i * 100 + 1, -1.0f, 1.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(blocks_D[i], size_D, 0.0f);
  }
  fill<__nv_bfloat16, FillMode::CONSTANT>(block_D_ref, size_D, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute reference GEMM
  reference_gemm<__nv_bfloat16, __nv_bfloat16>(block_D_ref, blocks_A[0], blocks_B[0], M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtGemm gemm;
  gemm.init(M, N, K);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_D[idx], stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < profiling_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_D[idx], stream);
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
  fill<__nv_bfloat16, FillMode::CONSTANT>(blocks_D[0], size_D, 0.0f);
  gemm.run(blocks_A[0], blocks_B[0], blocks_D[0], stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  check_correctness(blocks_D[0], block_D_ref, size_D);

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  gemm.destroy();

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaFree(blocks_A[i]));
    CHECK_CUDA(cudaFree(blocks_B[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
  CHECK_CUDA(cudaFree(block_D_ref));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "cuBLASLt BF16 GEMM Profiler" << std::endl;
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
