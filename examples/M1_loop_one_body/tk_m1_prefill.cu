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
void prefill_ker(
        int CS, int HF,
        T* __W1,
        const T* __XA, const T* __XB, const T* __XC,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x*(HF*HF);
    const H *_XA       = reinterpret_cast<const H*>(__XA) + blockIdx.x*(CS*HF);
    const H *_XB       = reinterpret_cast<const H*>(__XB) + blockIdx.x*(CS*HF);
    const H *_XC       = reinterpret_cast<const H*>(__XC) + blockIdx.x*(CS*HF);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x*(CS*HF);

    /*********
    REGISTER
    **********/
    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    rt_bf<1, 4> XA_reg;
    rt_bf<1, 4> XB_reg;
    rt_bf<1, 4> XC_reg;

    rt_fl<1, 4> Z1_fl_reg;
    rt_bf<1, 4> Z1_reg;

    rt_bf<1, 4> Output_reg;
    rt_fl<1, 4> Z1_bar_term_1_fl_reg;
    rt_bf<1, 4> Z1_bar_term_1_reg;
    rt_fl<1, 4> Z1_bar_term_2_fl_reg;
    rt_bf<1, 4> Z1_bar_term_2_reg;
    rt_fl<1, 1> Attn1_fl_reg;
    rt_bf<1, 1> Attn1_reg;


    load(W1_reg, _W1, W1_reg.cols);
    load(XB_reg, _XB, XB_reg.cols);
    load(XA_reg, _XA, XA_reg.cols);
    load(XC_reg, _XC, XC_reg.cols);

    zero(Z1_fl_reg);
    mma_AB(Z1_fl_reg, XB_reg, W1_reg, Z1_fl_reg); // [K,f] r, [f,f] c -> [K,f] r

    copy(Z1_reg, Z1_fl_reg);
    sub(Z1_reg, Z1_reg, XA_reg);

    rt_bf<1, 4, ducks::rt_layout::col> &Z1_col_reg = swap_layout_inplace(Z1_reg); // row-maj -> col-maj

    zero(Attn1_fl_reg);
    mma_ABt(Attn1_fl_reg, XC_reg, XB_reg, Attn1_fl_reg);  // [N,K] r, [M,K] r -> [N,M] r
    copy(Attn1_reg, Attn1_fl_reg);
    make_causal(Attn1_reg, Attn1_reg, base_types::constants<bf16>::zero());

    zero(Z1_bar_term_1_fl_reg);
    mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_reg, Z1_bar_term_1_fl_reg); // [N,K] r, [K,M] c -> [N,M] r
    copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

    zero(Z1_bar_term_2_fl_reg);
    mma_AB(Z1_bar_term_2_fl_reg, Attn1_reg, Z1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K] r, [K,f] c -> [K,f] r
    copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg);

    sub(Output_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

    store(_Output, Output_reg, Output_reg.cols);

    rt_bf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);

    rt_fl<4, 4> W1_fl_reg;
    zero(W1_fl_reg);
    mma_AtB(W1_fl_reg, XB_col_reg, Z1_col_reg, W1_fl_reg);

    rt_bf<4, 4> W1_row_reg;
    copy(W1_row_reg, W1_fl_reg);
    rt_bf<4, 4, kittens::ducks::rt_layout::col> &W1_col_reg = swap_layout_inplace(W1_row_reg);

    sub(W1_reg, W1_reg, W1_col_reg);

    store(_W1, W1_reg, W1_reg.cols);

}

void
prefill
(
    torch::Tensor W1,
    torch::Tensor XA,
    torch::Tensor XB,
    torch::Tensor XC,
    torch::Tensor Output,
    cudaStream_t stream
) {

    auto batch_mul_head = XA.size(0);
    auto cs    = XA.size(1);
    auto hf    = XA.size(2);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_ker<H,T><<<batch_mul_head, threads, 0, stream>>>(
            cs, hf,
            W1.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
