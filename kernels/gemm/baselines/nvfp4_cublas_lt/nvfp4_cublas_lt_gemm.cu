/***************************************************************************************************
 * cuBLASLt NVFP4 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * D[m,n] = sum_k A[m,k] * B[n,k]
 * Input: FP4 E2M1 with block-wise UE4M3 scaling (16-element blocks along K) + per-tensor FP32 scale
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

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
// cuBLASLt NVFP4 GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Block-wise UE4M3 scaling with 16-element blocks + per-tensor FP32 scale
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtNvfp4Gemm {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;
  void* workspace;
  size_t workspaceSize;

  void init(int M, int N, int K) {
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matmul descriptor with FP32 compute
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // NVFP4 GEMM requires TN layout
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Set NVFP4 block scaling mode (VEC16_UE4M3 = 16-element blocks with UE4M3 scales)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Layout for B (cuBLAS "A"): FP4 packed
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, N, K));
    // Layout for A (cuBLAS "B"): FP4 packed
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // Workspace
    workspaceSize = 128 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));

    // Set dummy scale pointers for heuristic selection (actual values don't affect algorithm choice)
    void* dummyScalePtr = workspace;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummyScalePtr, sizeof(dummyScalePtr)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummyScalePtr, sizeof(dummyScalePtr)));

    // Get best algorithm once during init (heuristic doesn't depend on scale pointer values)
    int returnedResults = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
                                                 preference, 1, &heuristic, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
      std::cerr << "No algorithm found! Status: " << status << ", results: " << returnedResults << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void run(__nv_fp4x2_e2m1 const* A, __nv_fp4x2_e2m1 const* B,
           __nv_fp8_e4m3 const* A_scale, __nv_fp8_e4m3 const* B_scale,
           float const* A_scale_global, float const* B_scale_global,
           __nv_bfloat16* D, cudaStream_t stream = nullptr) {

    // Set block scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &B_scale, sizeof(B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &A_scale, sizeof(A_scale)));

    // Per-tensor scales are incorporated into alpha
    float h_a_scale, h_b_scale;
    cudaMemcpy(&h_a_scale, A_scale_global, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b_scale, B_scale_global, sizeof(float), cudaMemcpyDeviceToHost);
    float alpha = h_a_scale * h_b_scale;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, &alpha,
                                 B, layoutA,
                                 A, layoutB,
                                 &beta,
                                 D, layoutC,
                                 D, layoutD,
                                 &heuristic.algo, workspace, workspaceSize, stream));
  }

  void destroy() {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
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
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // L2 cache eviction - multiple buffer groups
  int l2_cache_size;
  cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);

  int K_blocks = K / 16;  // NVFP4 block size
  size_t size_A_packed = size_t(M) * K / 2;  // FP4 packed 2 per byte
  size_t size_B_packed = size_t(N) * K / 2;
  size_t size_A_scale = size_t(M) * K_blocks;
  size_t size_B_scale = size_t(N) * K_blocks;
  size_t size_D = size_t(M) * N;

  const size_t arg_size = size_A_packed + size_B_packed +
                          size_A_scale + size_B_scale +
                          2 * sizeof(float) +
                          2 * size_D * sizeof(__nv_bfloat16);
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

  // Allocate buffer groups
  std::vector<__nv_fp4x2_e2m1*> blocks_A(arg_group_count);
  std::vector<__nv_fp4x2_e2m1*> blocks_B(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_A_scale(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_B_scale(arg_group_count);
  std::vector<float*> blocks_A_scale_global(arg_group_count);
  std::vector<float*> blocks_B_scale_global(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_D(arg_group_count);
  __nv_bfloat16* block_D_ref;

  CHECK_CUDA(cudaMalloc(&block_D_ref, size_D * sizeof(__nv_bfloat16)));

  uint64_t seed = 2024;

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A_packed * sizeof(__nv_fp4x2_e2m1)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B_packed * sizeof(__nv_fp4x2_e2m1)));
    CHECK_CUDA(cudaMalloc(&blocks_A_scale[i], size_A_scale * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_B_scale[i], size_B_scale * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_A_scale_global[i], sizeof(float)));
    CHECK_CUDA(cudaMalloc(&blocks_B_scale_global[i], sizeof(float)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(__nv_bfloat16)));

    fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(blocks_A[i]), size_A_packed, seed + i * 100, 0.0f, 255.0f);
    fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(blocks_B[i]), size_B_packed, seed + i * 100 + 1, 0.0f, 255.0f);
    fill<__nv_fp8_e4m3, FillMode::RANDOM>(blocks_A_scale[i], size_A_scale, seed + i * 100 + 2, 0.1f, 10.0f);
    fill<__nv_fp8_e4m3, FillMode::RANDOM>(blocks_B_scale[i], size_B_scale, seed + i * 100 + 3, 0.1f, 10.0f);
    fill<float, FillMode::RANDOM>(blocks_A_scale_global[i], 1, seed + i * 100 + 4, 0.1f, 10.0f);
    fill<float, FillMode::RANDOM>(blocks_B_scale_global[i], 1, seed + i * 100 + 5, 0.1f, 10.0f);
    fill<__nv_bfloat16, FillMode::CONSTANT>(blocks_D[i], size_D, 0.0f);
  }
  fill<__nv_bfloat16, FillMode::CONSTANT>(block_D_ref, size_D, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute reference GEMM
  reference_nvfp4_gemm<__nv_bfloat16>(
      block_D_ref, blocks_A[0], blocks_B[0],
      blocks_A_scale[0], blocks_B_scale[0],
      blocks_A_scale_global[0], blocks_B_scale_global[0],
      M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtNvfp4Gemm gemm;
  gemm.init(M, N, K);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx],
             blocks_A_scale[idx], blocks_B_scale[idx],
             blocks_A_scale_global[idx], blocks_B_scale_global[idx],
             blocks_D[idx], stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < profiling_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx],
             blocks_A_scale[idx], blocks_B_scale[idx],
             blocks_A_scale_global[idx], blocks_B_scale_global[idx],
             blocks_D[idx], stream);
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
  gemm.run(blocks_A[0], blocks_B[0],
           blocks_A_scale[0], blocks_B_scale[0],
           blocks_A_scale_global[0], blocks_B_scale_global[0],
           blocks_D[0], stream);
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
    CHECK_CUDA(cudaFree(blocks_A_scale[i]));
    CHECK_CUDA(cudaFree(blocks_B_scale[i]));
    CHECK_CUDA(cudaFree(blocks_A_scale_global[i]));
    CHECK_CUDA(cudaFree(blocks_B_scale_global[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
  CHECK_CUDA(cudaFree(block_D_ref));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  std::cout << "cuBLASLt NVFP4 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "D[m,n] = sum_k A[m,k] * B[n,k]" << std::endl;
  std::cout << "Input: FP4 E2M1 + UE4M3 block scales (16-element blocks along K) + FP32 per-tensor scale" << std::endl;
  std::cout << "Accumulator: FP32, Output: BF16" << std::endl;
  std::cout << "Warmup: " << warmup_iters << ", Profiling: " << profiling_iters << std::endl;

  benchmark(1024, 1024, 1024);
  benchmark(2048, 2048, 2048);
  benchmark(4096, 4096, 4096);
  benchmark(8192, 8192, 8192);
  benchmark(16384, 16384, 16384);

  return 0;
}
