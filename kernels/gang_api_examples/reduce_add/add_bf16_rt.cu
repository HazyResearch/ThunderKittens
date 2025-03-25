#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 32;

using namespace kittens;

// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using pgl_m  =  pgl_manager<gl<bf16, 1, 1, -1, -1>, true>;
using kittens_pgl = kittens::pgl<global_layout>;
using rt_tile = kittens::rt<bf16, 16, 16>;
using st_tile = kittens::st<bf16, 16, 16>;

__global__ void all_reduce_int(kittens_pgl p_o) {
    // rt_tile tile;
    // kittens::load(tile, p_o.gl, {0, 0});
    // kittens::one(tile);
    // kittens::atomic_add(p_o, tile, {0, 1});
    
    st_tile tile; 
    kittens::load(tile, p_o.gl, {0, 0});
    kittens::one(tile);
    kittens::atomic_add(p_o, tile, {0, 1});
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Use float for host arrays and convert to/from bf16 during transfer
    float *host_mat_1_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1_float[i] = 0.0f;

    float *host_mat_2_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2_float[i] = 0.0f;

    // Allocate host bf16 arrays for data transfer
    bf16 *host_mat_1 = new bf16[nelem];
    bf16 *host_mat_2 = new bf16[nelem];

    // Convert from float to bf16 for device transfer
    for (int i = 0; i < nelem; ++i) {
        host_mat_1[i] = __float2bfloat16(host_mat_1_float[i]);
        host_mat_2[i] = __float2bfloat16(host_mat_2_float[i]);
    }

    // Print initial data (converting back to float for printing)
    // printf("Device 1: \n");
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", __bfloat162float(host_mat_1[i * N + j]));
    //     }
    //     printf("\n");
    // }   
    // printf("\n\n");

    // printf("Device 2: \n");
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         printf("%f ", __bfloat162float(host_mat_2[i * N + j]));
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");
    
    // Allocate and copy data to device
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    
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
    
    // Convert from bf16 to float for printing results
    for (int i = 0; i < nelem; ++i) {
        host_mat_1_float[i] = __bfloat162float(host_mat_1[i]);
        host_mat_2_float[i] = __bfloat162float(host_mat_2[i]);
    }
    
    // Print results
    printf("Device 1: \n");
    for (int i = 0; i < N; ++i) {
        if (i % 16 == 0 && i != 0) printf("\n");
        for (int j = 0; j < N; ++j) {
            if (j % 16 == 0 && j != 0) printf(" ");
            printf("%f ", host_mat_1_float[i * N + j]);
        }
        printf("\n");
    }   
    printf("\n\n");

    printf("Device 2: \n");
    for (int i = 0; i < N; ++i) {
        if (i % 16 == 0 && i != 0) printf("\n");
        for (int j = 0; j < N; ++j) {
            if (j % 16 == 0 && j != 0) printf(" ");
            printf("%f ", host_mat_2_float[i * N + j]);
        }
        printf("\n");
    }
    printf("\n\n");

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