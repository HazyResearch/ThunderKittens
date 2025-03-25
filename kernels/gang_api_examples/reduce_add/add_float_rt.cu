#include "kittens.cuh"

#include <random>
// #include <cuda_bf16.h>   // No longer needed for float

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 32;

using namespace kittens;

// Change bf16 to float in the layout definitions
using global_layout   = gl<float, 1, 1, -1, -1>;
using pgl_m  = pgl_manager<gl<float, 1, 1, -1, -1>, true>;
using kittens_pgl     = kittens::pgl<global_layout>;
using rt_tile         = kittens::rt<float, 32, 32>;

__global__ void all_reduce_int(kittens_pgl p_o) {
    rt_tile tile;
    kittens::load(tile, p_o.gl, {0, 0});
    kittens::one(tile);
    kittens::broadcast(p_o, tile, {0, 0});
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(float);

    // Create host arrays in float (initially zeroed)
    float *host_mat_1 = new float[nelem];
    float *host_mat_2 = new float[nelem];
    for (int i = 0; i < nelem; ++i) {
        host_mat_1[i] = 0.0f;
        host_mat_2[i] = 0.0f;
    }

    // Optionally, keep separate arrays for printing results
    float *host_mat_1_print = new float[nelem];
    float *host_mat_2_print = new float[nelem];

    // Allocate and copy data to device
    float **dev_mats = new float*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        device_ids[i] = i;
    }
    
    cudaSetDevice(0);
    pglCudaMalloc(NUM_DEVICES, device_ids, 0, &dev_mats[0], &dev_handles[0], size);
    cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    pglCudaMalloc(NUM_DEVICES, device_ids, 1, &dev_mats[1], &dev_handles[1], size);
    cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice);

    // Initialize parallel global layout
    pgl_m dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    dim3 grid(1);
    dim3 block(32);
    cudaSetDevice(0);
    all_reduce_int<<<grid, block>>>(dev_mat_pgl.get_pgl_obj(0));
    printf("Device 0 mc_ptr: %p\n", dev_mat_pgl.get_pgl_obj(0).mc_ptr);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);

    // For printing (data is already float, so simply copy)
    for (int i = 0; i < nelem; ++i) {
        host_mat_1_print[i] = host_mat_1[i];
        host_mat_2_print[i] = host_mat_2[i];
    }
    
    // (Optional) Print results (currently commented out)
    // printf("Device 1: \n");
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", host_mat_1_print[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");
    //
    // printf("Device 2: \n");
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", host_mat_2_print[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

    // Check correctness, should be all ones
    for (int i = 0; i < nelem; ++i) {
        if (host_mat_1_print[i] != 1.0f) {
            std::cerr << "Error: Device 1 element " << i << " is " << host_mat_1_print[i] << std::endl;
            return 1;
        }
        if (host_mat_2_print[i] != 1.0f) {
            std::cerr << "Error: Device 2 element " << i << " is " << host_mat_2_print[i] << std::endl;
            return 1;
        }
    }
    printf("Results are correct!\n");

    // Cleanup and exit
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;
    delete[] host_mat_1_print;
    delete[] host_mat_2_print;

    std::cout << "Done!" << std::endl;
    return 0;
}
