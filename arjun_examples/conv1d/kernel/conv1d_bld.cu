// Use C++ harness for now
//#define TORCH_COMPILE

// TODO this doesn't work b/c we'd have to loop thru every single row index
// Or create a set amount of blocks and have each block attend to a subset of the rows
// Then we'd loop by row within the block
// In addition to the looping by batch that we do

#include "src/kittens.cuh"

// Using Toeplitz matrix approach for short conv

using namespace kittens;

#define MMA_K 16
#define MMA_N 16
#define MMA_D 16

// Warps per block, divide the batches equally amongst them
#define NUM_WORKERS 4

//const int d_model = 16;

__global__ void short_conv(bf16 const* weights, bf16 const* u, bf16 *out, 
                            size_t k, size_t n, size_t d, size_t b) {
    
    // Use FULL mode for Toeplitz matrix - user specified padding
    // Make warp the batch dimension, grid dim is L * D
    
    // Number of tiles in row of conv matrix and col of input vec
    size_t N_tiles = (n + MMA_N - 1) / MMA_N;

    // How many batches each warp handles
    size_t warp_batches = (b + NUM_WORKERS - 1) / NUM_WORKERS;
    // Batch index of input
    size_t batch_group_idx = kittens::warpid() * warp_batches;
    // Channel index of input, slice of conv tensor
    size_t channel_idx = blockIdx.x * MMA_D;
    // Row index of given slice of conv tensor
    size_t row_idx = blockIdx.y * MMA_K;

    // if (row_idx >= k || batch_group_idx >= b || channel_idx >= d) {
    //     return;
    // }
    // Float register to do accumulate, but later convert back to bf16
    kittens::rt_fl<1,1> o_reg_mma;
    kittens::rt_bf<1,1> o_reg;
    kittens::rt_bf<1,1> h_reg;
    kittens::rt_bf<1,1,kittens::ducks::rt_layout::col> x_reg;

    // Iterate thru each batch in the group that the warp is assigned
    //#pragma unroll
    for (auto slice = 0; slice < warp_batches; slice++) {
        auto batch_idx = batch_group_idx + slice;
        kittens::zero(o_reg_mma);
        
        //#pragma unroll
        // Accumulate all row/col tiles in given block
        for (auto i = 0; i < N_tiles; i++) {
            // Slice offset + row offset + tile offset
            // We are tiling across model dim so row stride is slice len (k*n)
            kittens::load(h_reg, H + (channel_idx * k * n) + (row_idx * n) + (i * MMA_N), k*n);
            // Slice offset + column offset + tile offset
            // Here, we tile across model dim which is the row stride anyways
            kittens::load(x_reg, X + (batch_idx * n * d) + channel_idx + (i * MMA_N * d), d);
            kittens::mma_AB(o_reg_mma, h_reg, x_reg, o_reg_mma);
        }
        // Cast to bf16 for output
        kittens::copy(o_reg, o_reg_mma);
        kittens::store(O + (batch_idx * k * d) + channel_idx + (row_idx * d), o_reg, d);
        __syncthreads();
    }
}

void launch_short_conv(bf16 const* H, bf16 const* X, bf16 *O, 
                        size_t k, size_t n, size_t d, size_t b,
                        long mem_size,
                        cudaStream_t stream) {
    
    // Each warp gets its own set of batches (fixed num of warps)
    const dim3 block_dim{
        (unsigned int)(NUM_WORKERS * kittens::WARP_THREADS)
    };
    const dim3 grid_dim{
        // Slices along batch dim (3rd dim)
        (unsigned int)(d + MMA_D - 1) / MMA_D,
        (unsigned int)(k + MMA_K - 1) / MMA_K
    };

    cudaFuncSetAttribute(
        short_conv,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    short_conv<<<grid_dim, block_dim, mem_size>>>(H, X, O, k, n, d, b);
}

// For launching from PyTorch
#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void short_conv(
    const torch::Tensor h,
    const torch::Tensor x,
    torch::Tensor out,
    int K,
    int N,
    int C,
    int B,
) {
    CHECK_INPUT(h);
    CHECK_INPUT(x);
    
    TORCH_CHECK(x.size(0) == N, "x has length N");
    TORCH_CHECK(x.size(1) == d_model,           "x is d_model?");
    TORCH_CHECK(out.size(1) == d_model,         "out is d_model?");
    TORCH_CHECK(x.size(1) == kittens::TILE_DIM, "X needs to be 1 tile wide");
    TORCH_CHECK(h.size(0) == K, "Conv matrix rows is correct?");
    TORCH_CHECK(out.size(0) == h.size(0), "Out is correct size?");
    TORCH_CHECK(h.size(1) == x.size(0), "Conv matrix has same cols as input length?");
    TORCH_CHECK(h.size(1) % kittens::TILE_DIM == 0,        "sequence length is divisible by 16?");
    TORCH_CHECK(x.size(0) % kittens::TILE_DIM == 0,        "sequence length is divisible by 16?");
    TORCH_CHECK(out.size(0) % kittens::TILE_DIM == 0,      "sequence length is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *h_ptr               = h.data_ptr<c10::BFloat16>();
    c10::BFloat16 *x_ptr               = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr               = out.data_ptr<c10::BFloat16>();

    const bf16* h_bf           = reinterpret_cast<const bf16*>(h_ptr);
    const bf16* x_bf           = reinterpret_cast<const bf16*>(x_ptr);
          bf16* o_bf           = reinterpret_cast<bf16*>(o_ptr);

    unsigned long mem_size = 108000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_short_conv<__bf16>(
        h_bf, x_bf, o_bf
        K, N, mem_size, stream
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);
}
#else
#include "harness.impl"
#endif
