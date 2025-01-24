#include "kittens.cuh"
#include <iostream>

using namespace kittens;

__global__ void donothing() {
    auto all_tmem = allocate_tmem();
    // Allocate shared memory
    __shared__ uint32_t tmem_addr;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    __shared__ semaphore sem;
    init_semaphore(sem, 0, 1);

    // Allocate A, B tiles
    auto &a_smem = al.allocate<st_bf<64, 16>>();
    auto &b_smem = al.allocate<st_bf<16, 16>>();

    one(a_smem);
    zero(b_smem);
    __syncthreads();
    if(threadIdx.x == 0) {
        b_smem[{15, 0}] = 1.0f;
    }
    __syncthreads();
    auto d_tile = all_tmem.subtile<tmem<float, 64, 16>>(0, 0);
    mma<0,1>(d_tile, a_smem, b_smem, sem);
    __syncwarp();

    auto a_tile = all_tmem.subtile<tmem<bf16, 32, 16>>(0, 32);
    rt_bf<32, 16> reg_tile;
    store(a_tile, reg_tile); 
    __syncwarp();

    wait(sem, 0);

    auto d32_tile = all_tmem.subtile<tmem<float, 32, 16>>(0, 0);

    rt_fl<32, 16> d_reg;
    load(d_reg, d32_tile);
    
    printf("%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f;\n",
        threadIdx.x,
        d_reg.tiles[0][0].data[0].x, d_reg.tiles[0][0].data[0].y,
        d_reg.tiles[0][0].data[1].x, d_reg.tiles[0][0].data[1].y,
        d_reg.tiles[0][0].data[2].x, d_reg.tiles[0][0].data[2].y,
        d_reg.tiles[0][0].data[3].x, d_reg.tiles[0][0].data[3].y,
        d_reg.tiles[1][0].data[0].x, d_reg.tiles[1][0].data[0].y,
        d_reg.tiles[1][0].data[1].x, d_reg.tiles[1][0].data[1].y,
        d_reg.tiles[1][0].data[2].x, d_reg.tiles[1][0].data[2].y,
        d_reg.tiles[1][0].data[3].x, d_reg.tiles[1][0].data[3].y
    );
}

int main() {
    std::cout << "Hello, World! Launching kernel..." << std::endl;
    donothing<<<1, 32, 48000>>>();
    std::cout << "Kernel launched successfully!" << std::endl;
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Kernel finished successfully!" << std::endl;
    return 0;
}