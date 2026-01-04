/***************************************************************************************************
 * cuBLASLt MXFP8 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * D[m,n] = sum_k A[m,k] * B[n,k]
 * Input: FP8 E4M3 with block-wise E8M0 scaling (32-element blocks along K)
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

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
static constexpr int mxfp8_block_size = 32;  // MXFP8 uses 32-element blocks

///////////////////////////////////////////////////////////////////////////////////////////////////
// Simple reference GEMM: D = A * B with MXFP8 block scaling
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// D[m,n] = sum_k A[m,k] * B[n,k]
// Scales are per 32 elements along K (reduction axis):
//   A_scale: RowMajor M x (K/32), B_scale: RowMajor N x (K/32)
///////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float decode_ue8m0(__nv_fp8_e8m0 scale) {
  // UE8M0: unsigned 8-bit exponent, value = 2^(scale - 127)
  // E8M0 is just raw exponent bits, so cast to uint8_t
  return exp2f((float)(*reinterpret_cast<const uint8_t*>(&scale)) - 127.0f);
}

__global__ void reference_gemm_kernel(
    __nv_bfloat16* D,
    __nv_fp8_e4m3 const* A,  // RowMajor MxK: A[m,k] at m*K + k
    __nv_fp8_e4m3 const* B,  // ColMajor NxK (N rows of K elements): B[n,k] at n*K + k
    __nv_fp8_e8m0 const* A_scale, // RowMajor M x K_blocks: A_scale[m, kb] at m*K_blocks + kb
    __nv_fp8_e8m0 const* B_scale, // RowMajor N x K_blocks: B_scale[n, kb] at n*K_blocks + kb
    int M, int N, int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;  // m index
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // n index

  int K_blocks = K / mxfp8_block_size;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A is RowMajor MxK: A[m,k] at m*K + k
      int a_idx = row * K + k;
      // B is ColMajor NxK (N rows of K elements): B[n,k] at n*K + k
      int b_idx = col * K + k;

      // Scales blocked along K: every 32 elements along K share a scale
      int kb = k / mxfp8_block_size;
      // A_scale is RowMajor M x K_blocks: A_scale[m, kb] at m*K_blocks + kb
      int a_scale_idx = row * K_blocks + kb;
      // B_scale is RowMajor N x K_blocks: B_scale[n, kb] at n*K_blocks + kb
      int b_scale_idx = col * K_blocks + kb;

      float a_s = decode_ue8m0(A_scale[a_scale_idx]);
      float b_s = decode_ue8m0(B_scale[b_scale_idx]);

      float a = float(A[a_idx]) * a_s;
      float b = float(B[b_idx]) * b_s;
      acc += a * b;
    }
    // D is RowMajor MxN: D[m,n] at m*N + n
    D[row * N + col] = __float2bfloat16(acc);
  }
}

void launch_reference_gemm(__nv_bfloat16* D, __nv_fp8_e4m3 const* A,
                           __nv_fp8_e4m3 const* B,
                           __nv_fp8_e8m0 const* A_scale, __nv_fp8_e8m0 const* B_scale,
                           int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  reference_gemm_kernel<<<grid, block>>>(D, A, B, A_scale, B_scale, M, N, K);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization kernels
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init_random_fp8_kernel(__nv_fp8_e4m3* data, size_t count, uint64_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    uint64_t x = seed + idx;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    // Generate values in [-1, 1] range (will be scaled by block scales)
    float u = (float)(x >> 40) * (1.0f / 16777216.0f);
    float val = u * 2.0f - 1.0f;
    data[idx] = __nv_fp8_e4m3(val);
  }
}

__global__ void init_random_scale_kernel(__nv_fp8_e8m0* data, size_t count, uint64_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    uint64_t x = seed + idx;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    // UE8M0 scale: use range [120, 134] which gives scales ~2^-7 to 2^7
    uint8_t scale_val = 120 + (x % 15);
    reinterpret_cast<uint8_t*>(data)[idx] = scale_val;
  }
}

__global__ void init_uniform_scale_kernel(__nv_fp8_e8m0* data, size_t count, uint8_t value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    reinterpret_cast<uint8_t*>(data)[idx] = value;
  }
}

void init_random_fp8(__nv_fp8_e4m3* data, size_t count, uint64_t seed) {
  dim3 block(256);
  dim3 grid((count + 255) / 256);
  init_random_fp8_kernel<<<grid, block>>>(data, count, seed);
}

void init_random_scale(__nv_fp8_e8m0* data, size_t count, uint64_t seed) {
  dim3 block(256);
  dim3 grid((count + 255) / 256);
  init_random_scale_kernel<<<grid, block>>>(data, count, seed);
}

void init_uniform_scale(__nv_fp8_e8m0* data, size_t count, uint8_t value) {
  dim3 block(256);
  dim3 grid((count + 255) / 256);
  init_uniform_scale_kernel<<<grid, block>>>(data, count, value);
}

__global__ void fill_zero_kernel(__nv_bfloat16* data, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = __float2bfloat16(0.0f);
  }
}

void fill_zero(__nv_bfloat16* data, size_t count) {
  dim3 block(256);
  dim3 grid((count + 255) / 256);
  fill_zero_kernel<<<grid, block>>>(data, count);
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
    float err = std::abs(__bfloat162float(h_D[i]) - __bfloat162float(h_D_ref[i]));

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
// cuBLASLt MXFP8 GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Block-wise scaling with 32-element blocks
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtMxfp8Gemm {
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

    // D[m,n] = sum_k A[m,k] * B[n,k]
    // A: RowMajor MxK = ColMajor KxM, B: ColMajor NxK = RowMajor NxK = ColMajor KxN
    // D: RowMajor MxN = ColMajor NxM
    // In col-major: D' = B'^T * A' where B' is KxN, A' is KxM, D' is NxM
    cublasOperation_t transA = CUBLAS_OP_T;  // B' (KxN) transposed gives NxK
    cublasOperation_t transB = CUBLAS_OP_N;  // A' (KxM) as-is gives KxM
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Set MXFP8 block scaling mode (VEC32_UE8M0 = 32-element blocks with UE8M0 scales)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Layout for B (cuBLAS "A"): RowMajor NxK = ColMajor KxN, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    // Layout for A (cuBLAS "B"): RowMajor MxK = ColMajor KxM, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM, ld=N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // Workspace
    workspaceSize = 128 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));
  }

  void run(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
           __nv_fp8_e8m0 const* A_scale, __nv_fp8_e8m0 const* B_scale,
           __nv_bfloat16* D, cudaStream_t stream = nullptr) {

    // Set scale pointers for this run
    // cuBLAS "A" = our B, cuBLAS "B" = our A
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &B_scale, sizeof(B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &A_scale, sizeof(A_scale)));

    // Get best algorithm
    int returnedResults = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
                                                 preference, 1, &heuristic, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
      std::cerr << "No algorithm found! Status: " << status << ", results: " << returnedResults << std::endl;
      exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS "A" = our B, cuBLAS "B" = our A
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
  // FP8 data + scales + BF16 output
  size_t a_elements = size_t(M) * K;
  size_t b_elements = size_t(N) * K;
  const size_t arg_size = a_elements + b_elements +
                          (a_elements + 31) / 32 + (b_elements + 31) / 32 +
                          2 * size_t(M) * N;
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

  // Allocate buffer groups
  std::vector<__nv_fp8_e4m3*> blocks_A(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_B(arg_group_count);
  std::vector<__nv_fp8_e8m0*> blocks_A_scale(arg_group_count);
  std::vector<__nv_fp8_e8m0*> blocks_B_scale(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_D(arg_group_count);
  __nv_bfloat16* block_D_ref;

  int K_blocks = K / mxfp8_block_size;

  size_t size_A = size_t(M) * K;
  size_t size_B = size_t(N) * K;
  size_t size_A_scale = size_t(M) * K_blocks;    // M x (K/32) - scales blocked along K
  size_t size_B_scale = size_t(N) * K_blocks;    // N x (K/32) - scales blocked along K
  size_t size_D = size_t(M) * N;

  CHECK_CUDA(cudaMalloc(&block_D_ref, size_D * sizeof(__nv_bfloat16)));

  uint64_t seed = 2024;
  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_A_scale[i], size_A_scale * sizeof(__nv_fp8_e8m0)));
    CHECK_CUDA(cudaMalloc(&blocks_B_scale[i], size_B_scale * sizeof(__nv_fp8_e8m0)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(__nv_bfloat16)));

    init_random_fp8(blocks_A[i], size_A, seed + i * 100);
    init_random_fp8(blocks_B[i], size_B, seed + i * 100 + 1);
    // Use uniform scales = 127 (2^0 = 1.0) for testing
    init_uniform_scale(blocks_A_scale[i], size_A_scale, 127);
    init_uniform_scale(blocks_B_scale[i], size_B_scale, 127);
    fill_zero(blocks_D[i], size_D);
  }
  fill_zero(block_D_ref, size_D);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute reference GEMM
  launch_reference_gemm(block_D_ref, blocks_A[0], blocks_B[0],
                        blocks_A_scale[0], blocks_B_scale[0], M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtMxfp8Gemm gemm;
  gemm.init(M, N, K);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_A_scale[idx], blocks_B_scale[idx], blocks_D[idx], stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < profiling_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_A_scale[idx], blocks_B_scale[idx], blocks_D[idx], stream);
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
  gemm.run(blocks_A[0], blocks_B[0], blocks_A_scale[0], blocks_B_scale[0], blocks_D[0], stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  verify_gemm(blocks_D[0], block_D_ref, M, N);

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
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
  CHECK_CUDA(cudaFree(block_D_ref));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "cuBLASLt MXFP8 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "D[m,n] = sum_k A[m,k] * B[n,k]" << std::endl;
  std::cout << "Input: FP8 E4M3 + E8M0 block scales (32-element blocks along K)" << std::endl;
  std::cout << "Accumulator: FP32, Output: BF16" << std::endl;
  std::cout << "Warmup: " << warmup_iters << ", Profiling: " << profiling_iters << std::endl;

  benchmark(1024, 1024, 1024);
  benchmark(2048, 2048, 2048);
  benchmark(4096, 4096, 4096);
  benchmark(8192, 8192, 8192);
  benchmark(16384, 16384, 16384);

  return 0;
}
