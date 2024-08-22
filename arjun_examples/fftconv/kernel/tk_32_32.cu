//#define TORCH_COMPILE

#include "src/kittens.cuh"

using namespace kittens;

// In prod set both to 16
// Number of batches per SM
#define B_TILE 1
// Number of heads per SM
#define H_TILE 1

// Right now no warp tiling
#define NUM_WORKERS 1


__global__ void fftconv_tk(bf16 *u_real, bf16 *u_imag, bf16 *kf_real, bf16 *kf_imag, bf16 *f_real, bf16 *f_imag,
                            bf16 *finv_real, bf16 *finv_imag, bf16 *tw_real, bf16 *tw_imag, bf16 *twinv_real, bf16 *twinv_imag, 
                            bf16 *o, int n, int n1){
    // Every block loads same seq tile

    // Doing 32x32 tiles
    kittens::rt_cmplx_bf<2, 2> a_reg;
    // For transpositions of first matrix (but still in row-layout)
    kittens::rt_cmplx_bf<2, 2> a_tr;
    // Bc we're using HMMA instructions, we need an fl accum reg for MMA
    kittens::rt_cmplx_fl<2, 2> mma_reg;
    // Separate reg to hold accumulated X values (in bf)
    kittens::rt_cmplx_bf<2, 2> accum;
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::col> b_reg;
    
    kittens::load(a_reg, f_real, f_imag, n1, n1);
    // F^T
    kittens::transpose_sep(a_tr, a_reg);
    // X
    kittens::load(b_reg, u_real, u_imag, n1, n1);

    // X = F^T X
    kittens::zero(mma_reg);
    kittens::mma_AB(mma_reg, a_tr, b_reg, mma_reg);
    kittens::copy(accum, mma_reg);
    
    // Load tw into a_reg
    kittens::load(a_reg, tw_real, tw_imag, n1, n1);
    // X = X * tw
    kittens::mul(accum, accum, a_reg);

    // X = XF
    kittens::load(b_reg, f_real, f_imag, n1, n1);
    kittens::zero(mma_reg);
    kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
    kittens::copy(accum, mma_reg);

    // X = X * K_f^T
    kittens::load(a_reg, kf_real, kf_imag, n1, n1);
    kittens::mul(accum, accum, a_reg);

    // X = XFinv
    kittens::load(b_reg, finv_real, finv_imag, n1, n1);
    kittens::zero(mma_reg);
    kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
    kittens::copy(accum, mma_reg); 
    
    // X = X^T * twinv
    kittens::transpose_sep(a_tr, accum);
    kittens::load(a_reg, twinv_real, twinv_imag, n1, n1);
    kittens::mul(a_tr, a_tr, a_reg);

    // Y = XFinv
    kittens::load(b_reg, finv_real, finv_imag, n1, n1);
    kittens::zero(mma_reg);
    kittens::mma_AB(mma_reg, a_tr, b_reg, mma_reg);
    kittens::copy(accum, mma_reg);

    // // Write Y^T to HBM
    kittens::transpose_sep(a_tr, accum);
    kittens::store(o, a_tr.real, n1);
    
}

void launch_fftconv_tk(bf16 *u_real, bf16 *u_imag, bf16 *kf_real, bf16 *kf_imag, bf16 *f_real, bf16 *f_imag,
                            bf16 *finv_real, bf16 *finv_imag, bf16 *tw_real, bf16 *tw_imag, bf16 *twinv_real, bf16 *twinv_imag, 
                            bf16 *o, int b, int h, int n, int n1, long mem_size) {
    const dim3 block_dim{
        (unsigned int)(kittens::WARP_THREADS)
    };
    // Constant number of blocks in each grid dim
    // const dim3 grid_dim{
    //     (unsigned int)b / B_TILE,
    //     (unsigned int)h / H_TILE
    // };
    const dim3 grid_dim{
        1u
    };

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(u_real, u_imag, kf_real, kf_imag, f_real, f_imag, finv_real, finv_imag, 
                                                tw_real, tw_imag, twinv_real, twinv_imag, o, n, n1);
}

#include "harness.impl"