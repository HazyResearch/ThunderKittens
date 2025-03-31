#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 2;
constexpr int NUM_WARPS = 8;
constexpr size_t N = 8192;

using namespace kittens;

using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using kittens_pgl = kittens::pgl<global_layout, 2, true>;
using rt_tile = kittens::rt<bf16, 64, 64>;
using st_tile = kittens::st<bf16, 16, 32>;

__global__ void multimem_red_kernel(kittens_pgl p_o, int dev_id) {
    /*
    Warp level register tile example
    */
    int row = blockIdx.x;
    int col = (blockIdx.y * NUM_WARPS) + kittens::warpid();

    if ((row * 64) >= N || (col * 64) >= N) return;

    rt_tile tile;
    // kittens::one(tile);
    // kittens::atomic_add(p_o, tile, dev_id, {row, col});
    kittens::all_reduce_add(tile, p_o, dev_id, {row, col});
    kittens::broadcast(p_o, tile, dev_id, {row, col});
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and copy data to device
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    float *host_mat_1_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1_float[i] = 1.0f;
    bf16 *host_mat_1 = new bf16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1[i] = __float2bfloat16(host_mat_1_float[i]);

    float *host_mat_2_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2_float[i] = 0.0f;
    bf16 *host_mat_2 = new bf16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2[i] = __float2bfloat16(host_mat_2_float[i]);
    
    cudaSetDevice(0);
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 0, &dev_mats[0], size);
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice));

    cudaSetDevice(1);
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 1, &dev_mats[1], size);
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice));


    // Initialize parallel global layout
    kittens_pgl dev_mat_pgl{device_ids, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    dim3 block(NUM_WARPS * kittens::WARP_THREADS);
    dim3 grid(N / 64, N / 64);
    printf("Grid: %d %d, Block: %d %d\n", grid.x, grid.y, block.x, block.y);

    int NUM_PROFILE_ITERS = 50;
    cudaSetDevice(0);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        multimem_red_kernel<<<grid, block>>>(dev_mat_pgl, 0);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_PROFILE_ITERS;
    printf("Average time per kernel launch: %f seconds\n", avg_time);

    // Bring back data
    cudaSetDevice(0);
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);
    
    // Convert from bf16 to float for printing results
    for (int i = 0; i < nelem; ++i) {
        host_mat_1_float[i] = __bfloat162float(host_mat_1[i]);
        host_mat_2_float[i] = __bfloat162float(host_mat_2[i]);
    }

    // Check correctness, should be all ones
    // for (int i = 0; i < nelem; ++i) {
    //     if (host_mat_1_float[i] != 1.0f) {
    //         std::cerr << "Error: Device 1 element " << i << " is " << host_mat_1_float[i] << std::endl;
    //         return 1;
    //     }
    //     if (host_mat_2_float[i] != 1.0f) {
    //         std::cerr << "Error: Device 2 element " << i << " is " << host_mat_2_float[i] << std::endl;
    //         return 1;
    //     }
    // }
    // printf("Results are correct!\n");

    // Cleanup and exit
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;
    delete[] host_mat_1_float;
    delete[] host_mat_2_float;

    std::cout << "Done!" << std::endl;
    return 0;
}