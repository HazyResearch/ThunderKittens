#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << status << std::endl;                           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        CHECK_CUBLAS(cublasCreate(&h));
        CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
        return h;
    }();
    return handle;
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasGemmEx(
        get_cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, N,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

#include "launch.cu"
