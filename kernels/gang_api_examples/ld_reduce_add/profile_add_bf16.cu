#include "kittens.cuh"

#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

constexpr int NUM_DEVICES = 8;
// constexpr size_t N = 2ULL * 1024 * 1024 * 1024;
constexpr size_t N = 32;

constexpr int ITER_PER_THREAD = 32;
constexpr int MAX_VEC_SIZE = 16;

using namespace kittens;

using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using pgl_m  =  pgl_manager<gl<bf16, 1, 1, -1, -1>, true>;
using kittens_pgl = kittens::pgl<global_layout>;

using rt_tile = kittens::rt<bf16, 16, 16>;

// Kernel to initialize matrices with random values on device
__global__ void initialize_matrix(bf16 *mat, size_t n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    
    if (idx < n) {
        curand_init(seed, idx, 0, &state);
        float random_val = curand_uniform(&state) - 0.5f; // Range -0.5 to 0.5
        mat[idx] = __float2bfloat16(random_val);
    }
}

__global__ void all_reduce_bf16(kittens_pgl p_o) {
    // kittens::all_reduce_add(p_o);
    rt_tile tile; 
    
    kittens::all_reduce_add(p_o, tile, {0, 0});
}

int main() {
    // Setup
    size_t nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate device memory
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    // Initialize each device with random data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        pglCudaMalloc(NUM_DEVICES, device_ids, dev_idx, &dev_mats[dev_idx], &dev_handles[dev_idx], size);
        
        // Initialize directly on device
        dim3 initGrid((nelem + 255) / 256);
        dim3 initBlock(256);
        initialize_matrix<<<initGrid, initBlock>>>(dev_mats[dev_idx], nelem, 42 + dev_idx); // Different seed per device
        cudaDeviceSynchronize();
    }

    // Initialize parallel global layout
    pgl_m dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    int nelem_per_dev = nelem / NUM_DEVICES;
    constexpr int nelem_per_block = 256 * ITER_PER_THREAD * (MAX_VEC_SIZE / sizeof(__nv_bfloat16));

    dim3 grid((nelem_per_dev + nelem_per_block - 1) / nelem_per_block);
    dim3 block(256);

    constexpr int NUM_ITERS = 1;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
        club.execute([&](int worker_id) {
            all_reduce_bf16<<<grid, block>>>(dev_mat_pgl.get_pgl_obj(worker_id));
        });
        club.execute([&](int worker_id) {
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_ITERS;
    printf("Average time: %f ms\n", avg_time * 1000);

    // Cleanup and exit
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        pglCudaFree(dev_idx, dev_mats[dev_idx], dev_handles[dev_idx], size);
    }
    delete[] dev_mats;
    delete[] dev_handles;

    std::cout << "Done!" << std::endl;
    return 0;
}