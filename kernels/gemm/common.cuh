#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "kittens.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Fill kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

enum class FillMode { CONSTANT, RANDOM };

template <typename T, FillMode mode>
__global__ void fill_kernel(T* data, size_t count, uint64_t seed, float min_val = 0., float max_val = 0.) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val;
        if constexpr (mode == FillMode::CONSTANT) {
            val = min_val;
        } else if constexpr (mode == FillMode::RANDOM) {
            // Splitmix64 hash for uniform random bits
            uint64_t x = seed + idx;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            // Upper 24 bits to float in [0,1)
            float u = (float)(x >> 40) * (1.0f / 16777216.0f);
            // Scale to [min_val, max_val]
            val = u * (max_val - min_val) + min_val;
        }
        data[idx] = kittens::base_types::convertor<T, float>::convert(val);
    }
}

template <typename T, FillMode mode>
static inline void fill(T* data, size_t count, float value) {
    static_assert(mode == FillMode::CONSTANT, "Use fill<T, FillMode::CONSTANT> for constant fill");
    dim3 block(256);
    dim3 grid((count + 255) / 256);
    fill_kernel<T, mode><<<grid, block>>>(data, count, 0, value, 0.0f);
}

template <typename T, FillMode mode>
static inline void fill(T* data, size_t count, uint64_t seed, float min_val, float max_val) {
    static_assert(mode == FillMode::RANDOM, "Use fill<T, FillMode::RANDOM> for random fill");
    dim3 block(256);
    dim3 grid((count + 255) / 256);
    fill_kernel<T, mode><<<grid, block>>>(data, count, seed, min_val, max_val);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference GEMM: D = A * B
// A: RowMajor (M x K)
// B: RowMajor (K x N) when transpose_b=false, ColMajor (N x K) when transpose_b=true
// D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputT, typename OutputT, bool transpose_b>
__global__ void reference_gemm_kernel(OutputT* D, InputT const* A, InputT const* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = kittens::base_types::convertor<float, InputT>::convert(A[row * K + k]);
            float b;
            if constexpr (transpose_b)
                b = kittens::base_types::convertor<float, InputT>::convert(B[col * K + k]);
            else
                b = kittens::base_types::convertor<float, InputT>::convert(B[k * N + col]);
            acc += a * b;
        }
        D[row * N + col] = kittens::base_types::convertor<OutputT, float>::convert(acc);
    }
}

template <typename InputT, typename OutputT, bool transpose_b = true>
static inline void reference_gemm(OutputT* D, InputT const* A, InputT const* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    reference_gemm_kernel<InputT, OutputT, transpose_b><<<grid, block>>>(D, A, B, M, N, K);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Block-Scaled Reference GEMM: D = A * B with block scaling
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline int scale_swizzle_idx(int row, int k_block, int K_blocks) {
  int M_block = row / 128;
  int K_block_groups = K_blocks / 4;
  int K_block_group = k_block / 4;
  int row_in_32 = row % 32;
  int tile_in_block = (row / 32) % 4;
  int kb_in_block = k_block % 4;

  int block_base = (M_block * K_block_groups + K_block_group) * 512;
  int local_idx = row_in_32 * 16 + tile_in_block * 4 + kb_in_block;
  return block_base + local_idx;
}

template <typename InputT, typename ScaleT, typename OutputT, int BLOCK_SIZE>
__global__ void reference_blockscaled_gemm_kernel(
    OutputT* D,
    InputT const* A, InputT const* B,
    ScaleT const* A_scale, ScaleT const* B_scale,
    int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    int K_blocks = K / BLOCK_SIZE;

    for (int k = 0; k < K; ++k) {
      int a_idx = row * K + k;
      int b_idx = col * K + k;

      int k_block = k / BLOCK_SIZE;
      int a_scale_idx = scale_swizzle_idx(row, k_block, K_blocks);
      int b_scale_idx = scale_swizzle_idx(col, k_block, K_blocks);

      float a_s = kittens::base_types::convertor<float, ScaleT>::convert(A_scale[a_scale_idx]);
      float b_s = kittens::base_types::convertor<float, ScaleT>::convert(B_scale[b_scale_idx]);

      float a = kittens::base_types::convertor<float, InputT>::convert(A[a_idx]) * a_s;
      float b = kittens::base_types::convertor<float, InputT>::convert(B[b_idx]) * b_s;

      acc += a * b;
    }
    D[row * N + col] = kittens::base_types::convertor<OutputT, float>::convert(acc);
  }
}

template <typename InputT, typename ScaleT, typename OutputT, int BLOCK_SIZE>
static inline void reference_blockscaled_gemm(
    OutputT* D,
    InputT const* A, InputT const* B,
    ScaleT const* A_scale, ScaleT const* B_scale,
    int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  reference_blockscaled_gemm_kernel<InputT, ScaleT, OutputT, BLOCK_SIZE>
      <<<grid, block>>>(D, A, B, A_scale, B_scale, M, N, K);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// NVFP4 Reference GEMM: D = A * B with packed FP4 data, block scales, and per-tensor scales
// A: RowMajor (M x K) packed as Mx(K/2), B: ColMajor (N x K) packed as Nx(K/2), D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename OutputT>
__global__ void reference_nvfp4_gemm_kernel(
    OutputT* D,
    __nv_fp4x2_e2m1 const* A_packed,
    __nv_fp4x2_e2m1 const* B_packed,
    __nv_fp8_e4m3 const* A_scale,
    __nv_fp8_e4m3 const* B_scale,
    float const* A_scale_global,
    float const* B_scale_global,
    int M, int N, int K) {

  constexpr int BLOCK_SIZE = 16;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    int K_blocks = K / BLOCK_SIZE;

    for (int k = 0; k < K; k += 2) {
      int a_idx = row * (K / 2) + k / 2;
      int b_idx = col * (K / 2) + k / 2;

      float2 a_vals = static_cast<float2>(A_packed[a_idx]);
      float2 b_vals = static_cast<float2>(B_packed[b_idx]);

      int k_block0 = k / BLOCK_SIZE;
      int k_block1 = (k + 1) / BLOCK_SIZE;

      int a_scale_idx0 = scale_swizzle_idx(row, k_block0, K_blocks);
      int a_scale_idx1 = scale_swizzle_idx(row, k_block1, K_blocks);
      int b_scale_idx0 = scale_swizzle_idx(col, k_block0, K_blocks);
      int b_scale_idx1 = scale_swizzle_idx(col, k_block1, K_blocks);

      float a_s0 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(A_scale[a_scale_idx0]);
      float a_s1 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(A_scale[a_scale_idx1]);
      float b_s0 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(B_scale[b_scale_idx0]);
      float b_s1 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(B_scale[b_scale_idx1]);

      acc += (a_vals.x * a_s0) * (b_vals.x * b_s0);
      acc += (a_vals.y * a_s1) * (b_vals.y * b_s1);
    }

    float global_scale = (*A_scale_global) * (*B_scale_global);
    D[row * N + col] = kittens::base_types::convertor<OutputT, float>::convert(acc * global_scale);
  }
}

template <typename OutputT>
static inline void reference_nvfp4_gemm(
    OutputT* D,
    __nv_fp4x2_e2m1 const* A, __nv_fp4x2_e2m1 const* B,
    __nv_fp8_e4m3 const* A_scale, __nv_fp8_e4m3 const* B_scale,
    float const* A_scale_global, float const* B_scale_global,
    int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  reference_nvfp4_gemm_kernel<OutputT>
      <<<grid, block>>>(D, A, B, A_scale, B_scale, A_scale_global, B_scale_global, M, N, K);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Correctness Check
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline void check_correctness(T const* d_out, T const* d_out_ref, size_t count) {
    std::vector<T> h_out(count);
    std::vector<T> h_out_ref(count);

    cudaMemcpy(h_out.data(), d_out, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_ref.data(), d_out_ref, count * sizeof(T), cudaMemcpyDeviceToHost);

    double abs_sum = 0.0, abs_max = 0.0;
    double err_sum = 0.0, err_max = 0.0;

    for (size_t i = 0; i < count; ++i) {
        float val = kittens::base_types::convertor<float, T>::convert(h_out[i]);
        float ref = kittens::base_types::convertor<float, T>::convert(h_out_ref[i]);
        float err = std::abs(val - ref);

        abs_sum += std::abs(val);
        abs_max = std::max(abs_max, (double)std::abs(val));
        err_sum += err;
        err_max = std::max(err_max, (double)err);
    }

    double abs_mean = abs_sum / count;
    double err_mean = err_sum / count;

    std::cout << "abs mean: " << std::setw(12) << abs_mean << std::endl;
    std::cout << "abs max:  " << std::setw(12) << abs_max << std::endl;
    std::cout << "err mean: " << std::setw(12) << err_mean << std::endl;
    std::cout << "err max:  " << std::setw(12) << err_max << std::endl;
}
