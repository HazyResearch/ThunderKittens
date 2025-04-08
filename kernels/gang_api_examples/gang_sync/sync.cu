#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>

#include "kittens.cuh"

constexpr int NUM_DEVICES = 8;
constexpr int NUM_BLOCKS = 1024;
constexpr int NUM_SYNC_POINTS = 16;
constexpr int PROFILE_ITERS = 50;

using namespace kittens;
using SyncManager = sync_manager<NUM_DEVICES, NUM_BLOCKS, NUM_SYNC_POINTS>;

__global__ void test_all_sync(SyncManager sm, int dev_idx) {
    using gang = kittens::gang<NUM_DEVICES>;
    gang::everyone::sync(sm, dev_idx);
}

__global__ void test_blockwise_sync(SyncManager sm, int dev_idx) {
    using gang = kittens::gang<NUM_DEVICES>;
    gang::blockwise::sync(sm, dev_idx);
}

__global__ void test_blockgroup_sync(SyncManager sm, int dev_idx) {
    using gang = kittens::gang<NUM_DEVICES>;
    gang::blockgroup::sync(sm, dev_idx, 0, NUM_BLOCKS * NUM_DEVICES);
}

// int num_blocks = gridDim.x * gridDim.y * gridDim.z * 4;

int main() {
    // Setup
    assert(NUM_DEVICES > 1); 
    CUCHECK(cuInit(0));    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    KittensClub club(device_ids, NUM_DEVICES);
    SyncManager sm = SyncManager::create(device_ids);

    // All sync test
    club.execute([&](int dev_idx) { // warmup
        test_all_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
        CUDACHECK(cudaDeviceSynchronize());
    });
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PROFILE_ITERS; iter++) {
        club.execute([&](int dev_idx) {
            test_all_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_ms = (elapsed.count() * 1e3) / PROFILE_ITERS;
    std::cout << "gang::everyone::sync (ms): " << avg_time_ms << std::endl;

    // Blockwise sync test
    club.execute([&](int dev_idx) { // warmup
        test_blockwise_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
        CUDACHECK(cudaDeviceSynchronize());
    });
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PROFILE_ITERS; iter++) {
        club.execute([&](int dev_idx) {
            test_blockwise_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    avg_time_ms = (elapsed.count() * 1e3) / PROFILE_ITERS;
    std::cout << "gang::blockwise::sync (ms): " << avg_time_ms << std::endl;

    // Blockgroup sync test
    club.execute([&](int dev_idx) { // warmup
        test_blockwise_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
        CUDACHECK(cudaDeviceSynchronize());
    });
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PROFILE_ITERS; iter++) {
        club.execute([&](int dev_idx) {
            test_blockwise_sync<<<NUM_BLOCKS, 256>>>(sm, dev_idx);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    avg_time_ms = (elapsed.count() * 1e3) / PROFILE_ITERS;
    std::cout << "gang::blockgroup::sync (ms): " << avg_time_ms << std::endl;

    // Cleanup
    sm.free();
    return 0;
}
