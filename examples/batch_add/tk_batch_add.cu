#include <iostream>
#include <math.h>
#include <assert.h>
//#include <mma_AB.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

# include "../../src/kittens.cuh"
# include "../../src/common/pyutils/torch_helpers.cuh"

// **** ASYNC INCLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace kittens;


template <typename H, typename T>
__global__
void batch_add_ker(
    const int CS, const int HF,
    const T* __XA, const T* __XB, T* __XC
) {

    const H *_XA       = reinterpret_cast<const H*>(__XA)+blockIdx.x*(CS*HF);
    const H *_XB       = reinterpret_cast<const H*>(__XB)+blockIdx.x*(CS*HF);
    H *_XC       = reinterpret_cast<H*>(__XC)+blockIdx.x*(CS*HF);

    rt_bf<1, 4> XA_reg;
    rt_bf<1, 4> XB_reg;
    rt_bf<1, 4> XC_reg;

    load(XA_reg, _XA, XA_reg.cols);
    load(XB_reg, _XB, XB_reg.cols);

    zero(XC_reg);
    add(XC_reg, XA_reg, XB_reg);

    store(_XC, XC_reg, XC_reg.cols);

}

void
batch_add
(
    torch::Tensor XA,
    torch::Tensor XB,
    torch::Tensor XC,
    cudaStream_t stream
) {

    auto batch_mul_head = XA.size(0);
    auto cs    = XA.size(1);
    auto hf    = XA.size(2);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    batch_add_ker<H,T><<<batch_mul_head, threads, 0, stream>>>(
        cs, hf,
        XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>()
    );

//    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
