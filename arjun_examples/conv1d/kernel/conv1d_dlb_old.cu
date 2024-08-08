// Use C++ harness for now

#include "src/kittens.cuh"

// Using Toeplitz matrix approach for short conv

using namespace kittens;

#define MMA_K 16
#define MMA_N 16
#define MMA_B 16

#define NUM_WORKERS 4

__global__ void short_conv(bf16 const* H, bf16 const* X, bf16 *O, 
                            size_t k, size_t n, size_t d, size_t b) {
    // Number of tiles in row of conv matrix and col of input vec
    auto N_tiles = (n + MMA_N - 1) / MMA_N;

    // How many batches each warp handles
    size_t warp_channels = (d + NUM_WORKERS - 1) / NUM_WORKERS;
    size_t channel_group_idx = kittens::warpid() * warp_channels;

    // Col index of input
    auto batch_idx = blockIdx.x * MMA_B;
    // Row index of given slice of conv tensor
    auto row_idx = blockIdx.y * MMA_K;

    if (row_idx >= k && batch_idx >= b && channel_group_idx >= c) {
        return;
    }
    // Float register to do accumulate, but later convert back to bf16
    kittens::rt_fl<1,1> o_reg_mma;
    kittens::rt_bf<1,1> o_reg;
    kittens::rt_bf<1,1> h_reg;
    kittens::rt_bf<1,1,kittens::ducks::rt_layout::col> x_reg;

    for (auto slice = 0; slice < warp_batches; slice++) {
        auto channel_idx = channel_group_idx + slice;
        kittens::zero(o_reg_mma);
        
        // Accumulate all row/col tiles in given block
        for (auto i = 0; i < N_tiles; i++) {
            // Slice offset + row offset + tile offset
            kittens::load(h_reg, H + (channel_idx * k * n) + (row_idx * n) + (i * MMA_N), n);
            // Slice offset + column offset + tile offset
            kittens::load(x_reg, X + (channel_idx * n * b) + batch_idx + (i * MMA_N * b), b);
            kittens::mma_AB(o_reg_mma, h_reg, x_reg, o_reg_mma);
        }
        // Cast to bf16 for output
        kittens::copy(o_reg, o_reg_mma);
        kittens::store(O + (channel_idx * k * b) + batch_idx + (row_idx * b), o_reg, b);
    }
}

void launch_short_conv(bf16 const* H, bf16 const* X, bf16 *O, 
                        size_t k, size_t n, size_t d, size_t b,
                        long mem_size,
                        cudaStream_t stream) {
    
    // Each warp gets a set of channels
    const dim3 block_dim{
        (unsigned int)(NUM_WORKERS * kittens::WARP_THREADS)
    };
    const dim3 grid_dim{
        // Slices along channel dim (3rd dim)
        (unsigned int)(b + MMA_B - 1) / MMA_B,
        (unsigned int)(k + MMA_K - 1) / MMA_K
    };

    cudaFuncSetAttribute(
        short_conv,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    short_conv<<<grid_dim, block_dim, mem_size>>>(H, X, O, k, n, c, b);
}

#include "harness.impl"
