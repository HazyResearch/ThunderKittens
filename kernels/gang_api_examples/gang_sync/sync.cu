#include <cuda_runtime.h>
#include <cuda/atomic>
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>

#include "kittens.cuh"

const int NUM_DEVICES = 8;

using namespace kittens;

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


__global__ void test_barrier_kernel(SyncSpace s) {
    using gang_1 = kittens::gang<0,1,2,3>;
    gang_1::sync(s); 
}

int main() {
    assert(NUM_DEVICES > 1); 

    CUCHECK(cuInit(0)); 
    
    std::vector<cudaStream_t> streams(NUM_DEVICES);
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaStreamCreate(&streams[dev_idx]));
    }

    /*
    Run kernel to profile barrier 
    */
    dim3 grid(256, 256, 1);
    dim3 block(256, 1, 1);
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    CUDACHECK(cudaSetDevice(0));
    KittensClub club(device_ids, NUM_DEVICES);
    club.execute([](int dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
    });
    
    const int PROFILE_ITERS = 50;

    SyncManager sync_m(NUM_DEVICES, device_ids);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PROFILE_ITERS; iter++) {
        club.execute([&](int dev_idx) {
            test_barrier_kernel<<<grid, block, 0, streams[dev_idx]>>>(sync_m.get_sync_space(dev_idx));
        });

        // Synchronize streams
        club.execute([&](int dev_idx) {
            CUDACHECK(cudaStreamSynchronize(streams[dev_idx]));
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_ms = (elapsed.count() * 1e3) / PROFILE_ITERS;

    std::cout << "Effective barrier overhead (ms): " << avg_time_ms << std::endl;

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaStreamDestroy(streams[dev_idx]));
    }

    return 0;
}
