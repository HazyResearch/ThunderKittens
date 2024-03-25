#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
using namespace nvcuda;

# include "src/kittens.cuh" // needs to come before torch_helpers, since torch_helpers relies on kittens.
#include "src/common/pyutils/torch_helpers.cuh"
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <ATen/cuda/CUDAContext.h>
#define NUM_WORKERS 8
using namespace kittens;

const int batch_size = 16;
const int K = 4096;
const int M = 14336;

template <typename H, typename T>
__global__
void batch_gemv_ker(const T* __input, const T* __weight, T* __output) {
    auto block_start = blockIdx.x;
    auto warpid = threadIdx.x / 32;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]);

    st_bf_1x4<ducks::st_layout::xor_swizzle> (&i_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    rt_bf_1x4<> i_reg, w_reg;
    rt_fl_1x1<> o_reg;

    const H *weight_g = device_cast(__weight) + (block_start * NUM_WORKERS + warpid) * w_reg.rows * K;
    const H *input_g = device_cast(__input);
    H *output_g = device_cast(__output) + (block_start * NUM_WORKERS + warpid) * o_reg.cols;

    zero(o_reg);
    for (int k_id = 0; k_id < K / i_reg.cols; k_id += NUM_WORKERS) {
        load(i_smem[warpid], input_g + (k_id + warpid) * i_reg.cols, K);
        __syncthreads();
        for (int subtile = 0; subtile < NUM_WORKERS; subtile++) {
            load(i_reg, i_smem[subtile]);
            load(w_reg, weight_g + (k_id + subtile) * w_reg.cols, K);
            dot(o_reg, i_reg, w_reg, o_reg);
            __syncthreads();
        }
    }
    store(output_g, o_reg, M);   
}

torch::Tensor 
batch_gemv(torch::Tensor input, torch::Tensor weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    assert(input.size(0) == batch_size);
    // output size is the input size with the last dimension replaced by the first dimension of weight size
    auto output = torch::empty({input.size(0), weight.size(0)}, input.options());
    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    auto threads = NUM_WORKERS * kittens::WARP_SIZE;
    auto stream_wrapper = at::cuda::getCurrentCUDAStream(input.device().index());
    cudaStream_t stream = stream_wrapper.stream();
    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(
        batch_gemv_ker<H, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    batch_gemv_ker<H, T><<<M / NUM_WORKERS / kittens::TILE_DIM, threads, mem_size, stream>>>(
        input.data_ptr<T>(),
        weight.data_ptr<T>(),
        output.data_ptr<T>()
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output;
}       