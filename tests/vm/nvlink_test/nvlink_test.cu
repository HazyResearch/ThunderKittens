#include <iostream>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime.h"

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

using namespace std;

constexpr int STRIDES = 32;
constexpr size_t STRIDE_SIZE = 10 * 1024 * 1024; // 10 MB
constexpr size_t GB = 1024 * 1024 * 1024;
__global__ void kernel(int* src, int* dst, int start) {
    int val = 0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (size_t i = 0; i < STRIDES; ++i) {
            volatile int _val;
            asm volatile (
                "{ ld.global.u32 %0, [%1]; }"
                : "=r"(_val)
                : "l"(&src[i + start * STRIDE_SIZE * STRIDES / 4])
                : "memory"
            );
            val += _val;
        }
        *dst = val;
    }
    __syncthreads();
}

void benchmark(int *src[2], int *dst[2], int src_dev, int dst_dev, int start, int warmup) {
    cudaEvent_t startEvent, stopEvent;
    CUDACHECK(cudaSetDevice(dst_dev));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
    CUDACHECK(cudaEventRecord(startEvent, 0));
    kernel<<<1, 1, 0, 0>>>(src[src_dev], dst[dst_dev], start); // Can't iter since this tests caching
    CUDACHECK(cudaEventRecord(stopEvent, 0));
    CUDACHECK(cudaEventSynchronize(stopEvent));
    float elapsedTimeMs;
    CUDACHECK(cudaEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    if (!warmup)
        cout << "D" << src_dev << " -> D" << dst_dev << ": " << 1e3 * elapsedTimeMs << " us" << endl;
}

int main() {
    // P2P Setup
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1));
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_1_0, 1, 0));
    cout << "Device 0 can access device 1: " << can_access_peer_0_1 << endl;
    cout << "Device 1 can access device 0: " << can_access_peer_1_0 << endl;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));

    // Allocate device memory (L2 cache on B200 is 126 MB)
    constexpr size_t SIZE = 2LL * GB;
    int *src[2];
    int *dst[2];
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc((void**)&src[0], SIZE));
    CUDACHECK(cudaMalloc((void**)&dst[0], sizeof(int)));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc((void**)&src[1], SIZE));
    CUDACHECK(cudaMalloc((void**)&dst[1], sizeof(int)));

    // Initialize to random values
    int *h_src[2];
    h_src[0] = new int[SIZE / sizeof(int)];
    h_src[1] = new int[SIZE / sizeof(int)];
    for (size_t i = 0; i < SIZE / sizeof(int); ++i) {
        h_src[0][i] = rand() % 100;
        h_src[1][i] = rand() % 100;
    }
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(src[0], h_src[0], SIZE, cudaMemcpyHostToDevice));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(src[1], h_src[1], SIZE, cudaMemcpyHostToDevice));
    delete[] h_src[0];
    delete[] h_src[1];
    
    // Launch benchmarks
    // void benchmark(int *src[2], int *dst[2], int src_dev, int dst_dev, int start, int warmup)
    benchmark(src, dst, 0, 0, 0, 1); // warmup
    benchmark(src, dst, 1, 1, 0, 1); // warmup
    benchmark(src, dst, 1, 0, 0, 1); // warmup
    benchmark(src, dst, 0, 1, 0, 1); // warmup
    benchmark(src, dst, 0, 0, 2, 0); // 0 -> 0
    benchmark(src, dst, 0, 0, 2, 0); // 0 -> 0
    benchmark(src, dst, 1, 0, 4, 0); // 1 -> 0
    benchmark(src, dst, 1, 0, 4, 0); // 1 -> 0
    benchmark(src, dst, 1, 1, 4, 0); // 1 -> 1
    benchmark(src, dst, 1, 1, 4, 0); // 1 -> 1

    // Cleanup
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaFree(src[0]));
    CUDACHECK(cudaFree(dst[0]));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaFree(src[1]));
    CUDACHECK(cudaFree(dst[1]));

    return 0;
}
