#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 32;

using namespace kittens;

// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;
// using pgl_m  =  pgl_manager<gl<bf16, 1, 1, -1, -1>, true>;
using kittens_pgl = kittens::pgl<global_layout, 2, true>;
using rt_tile = kittens::rt<bf16, 16, 32>;
using rv_vec = kittens::rv<bf16, 64>;
using st_tile = kittens::st<bf16, 16, 32>;

__global__ void all_reduce_int(kittens_pgl p_o, int dev_id) {
    /*
    Warp level register vector example
    */
    // if (warpid() != 0) return;
    // rv_vec vec;
    // kittens::one(vec);
    // kittens::broadcast(p_o, vec, dev_id, {0, 0});

    /*
    Warp level register tile example
    */
    if (warpid() != 0) return;
    rt_tile tile;
    kittens::one(tile);
    kittens::store(p_o[dev_id], tile, {0, 0});

    /*
    Group level register tile example
    */
    // using friends = kittens::group<2>;
    // rt_tile tile; 
    // kittens::one(tile);
    // friends::atomic_add(p_o, tile, dev_id, {0, friends::groupid()});    

    /*
    Warp level shared tile example 
    */
    // extern __shared__ kittens::alignment_dummy __shm[];
    // kittens::shared_allocator al((int*)&__shm[0]);
    // st_tile (&s_tile) = al.allocate<st_tile>();
    // warpgroup::one(s_tile);
    // __syncthreads();
    // if (kittens::warpid() == 0) {
    //     kittens::atomic_add(p_o, s_tile, dev_id, {dev_id, 0});
    // }

    /*
    Group level shared tile example
    */
    // using friends = kittens::group<2>;
    // extern __shared__ kittens::alignment_dummy __shm[];
    // kittens::shared_allocator al((int*)&__shm[0]);
    // st_tile (&s_tile)[2] = al.allocate<st_tile, 2>();
    // friends::one(s_tile[friends::groupid()]);
    // __syncthreads();
    // friends::atomic_add(p_o, s_tile[friends::groupid()], dev_id, {friends::groupid(), 0});
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
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 0, &dev_mats[0], size);
    cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 1, &dev_mats[1], size);
    cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice);

    // Initialize parallel global layout
    kittens_pgl dev_mat_pgl{device_ids, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);
    unsigned long smem = 2 * 32 * 32 * sizeof(bf16);

    dim3 grid(1);
    dim3 block(128);
    cudaSetDevice(0);
    all_reduce_int<<<grid, block, smem>>>(dev_mat_pgl, 0);
    // printf("Device 0 mc_ptr: %p\n", dev_mat_pgl.get_pgl_obj(0).mc_ptr);
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
        if (i == 8) printf("\n\n");
        for (int j = 0; j < N; ++j) {
            if (j % 16 == 0 && j != 0) printf(" ");
            printf("%f ", host_mat_1_float[i * N + j]);
        }
        printf("\n");
    }   
    printf("\n\n");

    // printf("Device 2: \n");
    // for (int i = 0; i < N; ++i) {
    //     if (i % 16 == 0 && i != 0) printf("\n");
    //     for (int j = 0; j < N; ++j) {
    //         if (j % 16 == 0 && j != 0) printf(" ");
    //         printf("%f ", host_mat_2_float[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

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