#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "kittens.cuh"

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
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputT, typename OutputT>
__global__ void reference_gemm_kernel(OutputT* D, InputT const* A, InputT const* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = kittens::base_types::convertor<float, InputT>::convert(A[row * K + k]);
            float b = kittens::base_types::convertor<float, InputT>::convert(B[col * K + k]);
            acc += a * b;
        }
        D[row * N + col] = kittens::base_types::convertor<OutputT, float>::convert(acc);
    }
}

template <typename InputT, typename OutputT>
static inline void reference_gemm(OutputT* D, InputT const* A, InputT const* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    reference_gemm_kernel<InputT, OutputT><<<grid, block>>>(D, A, B, M, N, K);
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
