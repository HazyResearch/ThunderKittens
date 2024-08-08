#define TORCH_COMPILE

#include "src/kittens.cuh"

// Using Toeplitz matrix approach for short conv

using namespace kittens;

#define MMA_K 16
#define MMA_N 16
#define MMA_B 16

#define NUM_WORKERS 4

__global__ void fftconv(){

}

void launch_conv1d(bf16 *w, long mem_size) {
    // Each warp gets a set of channels
    const dim3 block_dim{
        (unsigned int)
    };
    const dim3 grid_dim{
        (unsigned int)
        (unsigned int)
    };

    cudaFuncSetAttribute(
        fftconv,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv<<<grid_dim, block_dim, mem_size>>>();
}
