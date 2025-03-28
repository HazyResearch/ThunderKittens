#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 8;
constexpr int NUM_WARPS = 8;
constexpr size_t N = 4096;

using namespace kittens;

using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using kittens_pgl = kittens::pgl<global_layout, 2, true>;
using rt_tile = kittens::rt<bf16, 64, 64>;
using st_tile = kittens::st<bf16, 16, 32>;

/*
Warp level register tile example
*/
// rt_tile tile;
// kittens::one(tile);
// kittens::atomic_add(p_o, tile, dev_id, {0, 1});

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
__global__ void multimem_red_kernel(kittens_pgl pgl, int dev_id) {
    int rows_per_device = pgl[dev_id].rows() / NUM_DEVICES;
    int row_start = (rows_per_device) * dev_id;
    int warp_start = (rows_per_device / NUM_WARPS) * blockIdx.x;
    int row = row_start + warp_start;
    int col = NUM_WARPS * blockIdx.y;
    
    rt_tile r_tile; 
    kittens::one(r_tile);
    kittens::atomic_add(pgl, r_tile, dev_id, {row, col});
}

bool verify_all_ones(const bf16* data, size_t size) {
    bool verification_passed = true;
    
    #pragma omp parallel
    {
        bool thread_found_error = false;
        size_t error_index = 0;
        float error_value = 0.0f;
        
        #pragma omp for
        for (size_t i = 0; i < size; ++i) {
            float val = __bfloat162float(data[i]);
            if (val != 1.0f && !thread_found_error) {
                thread_found_error = true;
                error_index = i;
                error_value = val;
            }
        }
        
        if (thread_found_error) {
            #pragma omp critical
            {
                verification_passed = false;
                std::cout << "Verification failed at index " << error_index 
                          << ": value = " << error_value << std::endl;
            }
        }
    }
    
    return verification_passed;
}

int main() {
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and copy data to device
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        pglCudaMalloc<true>(NUM_DEVICES, device_ids, i, &dev_mats[i], &dev_handles[i], size);
        cudaMemset(dev_mats[i], 0, size);
    }

    // Initialize parallel global layout
    kittens_pgl dev_mat_pgl{device_ids, dev_mats, nullptr, nullptr, N, N};

    // Club initialiation
    KittensClub club(device_ids, NUM_DEVICES);
    club.execute([](int dev_idx) {
        cudaSetDevice(dev_idx);
    });

    unsigned long smem = 2 * 32 * 32 * sizeof(bf16);
    dim3 grid((N / NUM_DEVICES) / (NUM_WARPS * 32), N / (NUM_WARPS * 32));
    dim3 block(256);
    printf("Grid: (%d, %d)\n", grid.x, grid.y);
    printf("Block: (%d)\n", block.x);
    
    for (int i = 0; i < NUM_DEVICES; ++i) {
        club.execute([&](int dev_idx) {
            multimem_red_kernel<<<grid, block, smem>>>(dev_mat_pgl, dev_idx);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        });
    }

    // Bring back data
    bf16 *host_mat_1 = new bf16[nelem];
    bf16 *host_mat_2 = new bf16[nelem];
    cudaSetDevice(0);
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);

    // Verify results
    bool verification_passed = verify_all_ones(host_mat_1, nelem) && verify_all_ones(host_mat_2, nelem);
    if (verification_passed) {
        std::cout << "Verification passed!" << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }
    
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;

    std::cout << "Done!" << std::endl;
    return 0;
}