#define TORCH_COMPILE

#include "src/kittens.cuh"

// Using Toeplitz matrix approach for short conv

using namespace kittens;

#define MMA_K 16
#define MMA_N 16
#define MMA_B 16

#define NUM_WORKERS 4

__global__ void conv1d(bf16 const* H, bf16 const* X, bf16 *O, 
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

    if (row_idx >= k && batch_idx >= b && channel_group_idx >= d) {
        return;
    }
    // Float register to do accumulate, but later convert back to bf16
    kittens::rt_fl<1,1> o_reg_mma;
    kittens::rt_bf<1,1> o_reg;
    kittens::rt_bf<1,1> h_reg;
    kittens::rt_bf<1,1,kittens::ducks::rt_layout::col> x_reg;

    for (auto slice = 0; slice < warp_channels; slice++) {
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
        __syncthreads();
    }
}

void launch_conv1d(bf16 const* h, bf16 const* x, bf16 *o, 
                        size_t k, size_t n, size_t d, size_t b,
                        long mem_size) {

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
        conv1d,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    conv1d<<<grid_dim, block_dim, mem_size>>>(h, x, o, k, n, d, b);
}

// For launching from PyTorch
#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
using namespace torch::indexing;
torch::Tensor conv1d_tk(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    uint padding
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    namespace F = torch::nn::functional;
    // Left, right, top, bottom
    torch::Tensor padded_input = F::pad(input, F::PadFuncOptions({0, 0, padding, padding})).to(torch::kCUDA);

    int K = weight.size(1);
    int D = input.size(0);
    int B = input.size(2);

    TORCH_CHECK(K % 2 == 1, "Filter size must be odd");

    int L_in = padded_input.size(1);
    int L_out = (input.size(1) + 2 * padding - K + 1);
    
    //TORCH_CHECK(input.size(0) == N, "x has length N");
    TORCH_CHECK(weight.size(0) == D, "inputs and weights have same num channels?");
    TORCH_CHECK(bias.size(0) == weight.size(0), "weights and bias have same num channels?");
    TORCH_CHECK(weight.dtype() == bias.dtype(), "weight and bias must have same dtype");
    TORCH_CHECK(L_in % kittens::TILE_DIM == 0 && L_out % kittens::TILE_DIM == 0, "Matrix fits within tiles?");

    // DLB order for kernel
    torch::Tensor out = torch::empty({D, L_out, B}, input.options());

    // Create on GPU
    //torch::Tensor H = torch::empty({D, L_out, L_in}, input.options());
    //std::cout << "H correct dtype: " << std::boolalpha << (H.dtype() == torch::kBFloat16) << std::endl;

    int TOTAL_ELEMENTS_H = D * L_out * L_in;
    bf16 *h_ptr;
    cudaMalloc(&h_ptr, TOTAL_ELEMENTS_H * sizeof(bf16));

    bf16 *weight_ptr = reinterpret_cast<bf16*>(weight.data_ptr<c10::BFloat16>());

    // Construct valid Toeplitz matrix by channel
    for (int d = 0; d < D; d++) {
        for (int i = 0; i < L_out; i++) {
            for (int j = 0; j < L_in; j++) {
                int data_idx = d * (L_out * L_in) + i * (L_in) * j;
                std::cout << "H idx " << data_idx << std::endl;
                int weight_idx = i-j+K-1;
                if (weight_idx >= 0 && weight_idx < K) {
                    std::cout << "Weight idx " << weight_idx << std::endl;
                    h_ptr[data_idx] = weight_ptr[d*K + weight_idx];
                } else {
                    h_ptr[data_idx] = __float2bfloat16(0.0f);
                }
            }
        }
        std::cout << "Looping, Finished channel " << d << std::endl;
    }


    // convert to bf16
    //c10::BFloat16 *h_ptr               = H.data_ptr<c10::BFloat16>();
    c10::BFloat16 *x_ptr               = padded_input.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr               = out.data_ptr<c10::BFloat16>();

    // All of these already on GPU b/c PyTorch allocated them there
    //const bf16* h_bf           = reinterpret_cast<const bf16*>(h_ptr);
    const bf16* h_bf           = reinterpret_cast<const bf16*>(h_ptr);
    const bf16* x_bf           = reinterpret_cast<const bf16*>(x_ptr);
          bf16* o_bf           = reinterpret_cast<bf16*>(o_ptr);


    unsigned long mem_size = 108000;

    std::cout << "Invoking TK kernel" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "Created stream" << std::endl;
    launch_conv1d(
        h_bf, x_bf, o_bf,
        L_out, L_in, D, B, 
        mem_size
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);
    std::cout << "Finished invoking TK kernel" << std::endl;

    // This pointer was created by us, not PyTorch
    cudaFree(h_ptr);

    // Data in this tensor will be filled in
    return out;
}
#else
#include "harness.impl"
#endif
