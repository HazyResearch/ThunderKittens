//#define TORCH_COMPILE

#include "include/kittens.cuh"

using namespace kittens;

// Number of batches per SM
#define B_TILE 8
// Number of heads per SM
#define H_TILE 8

#define NUM_WORKERS 1


__global__ void fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const bf16 *kf_real, const bf16 *kf_imag, 
                            const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
                            const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
                            bf16 *o, 
                            int h, int n, int n1){
    int warpid = kittens::warpid();

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;
    // Doing 32x32 tiles
    kittens::rt_cmplx_bf<2, 2> a_reg;
    // For transpositions of first matrix (but still in row-layout)
    kittens::rt_cmplx_bf<2, 2> a_tr;
    // Bc we're using HMMA instructions, we need an fl accum reg for MMA
    kittens::rt_cmplx_fl<2, 2> mma_reg;
    // Separate reg to hold accumulated X values (in bf)
    kittens::rt_cmplx_bf<2, 2> accum;
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::col> b_reg;
    
    #pragma unroll
    for (int i = 0; i < H_TILE; i++) {
        int k_start = (h_start + i) * h_stride;
        #pragma unroll
        for (int j = 0; j < B_TILE; j++) {
            int u_start = ((b_start + j) * b_stride) + k_start;
            // X = F^T X
            kittens::load(a_reg, f_real, f_imag, n1, n1);
            kittens::transpose_sep(a_tr, a_reg);
            kittens::load(b_reg, u_real + u_start, u_imag + u_start, n1, n1);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, a_tr, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * tw
            kittens::load(a_reg, tw_real, tw_imag, n1, n1);
            kittens::mul(accum, accum, a_reg);
            // X = XF
            kittens::load(b_reg, f_real, f_imag, n1, n1);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * K_f^T
            kittens::load(a_reg, kf_real + k_start, kf_imag + k_start, n1, n1);
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
            kittens::store(o + u_start, a_tr.real, n1);
        }
    }
}

void launch_fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const bf16 *kf_real, const bf16 *kf_imag, 
                        const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
                        const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
                        bf16 *o, 
                        int b, int h, int n, int n1) {
    // 1 warp for 32x32 case
    const dim3 block_dim{
        (unsigned int)(kittens::WARP_THREADS)
    };
    const dim3 grid_dim{
        (unsigned int)(b + B_TILE - 1) / B_TILE,
        (unsigned int)(h + H_TILE - 1) / H_TILE
    };

    long mem_size = 1000;

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(u_real, u_imag, kf_real, kf_imag, 
                                                f_real, f_imag, finv_real, finv_imag, 
                                                tw_real, tw_imag, twinv_real, twinv_imag, 
                                                o, 
                                                h, n, n1);
}

// For launching from PyTorch
#ifdef TORCH_COMPILE
#include "include/common/pyutils/torch_helpers.cuh"
torch::Tensor fftconv_tk(
    const torch::Tensor u_real,
    const torch::Tensor u_imag,
    const torch::Tensor kf_real,
    const torch::Tensor kf_imag,
    const torch::Tensor f_real,
    const torch::Tensor f_imag,
    const torch::Tensor finv_real,
    const torch::Tensor finv_imag,
    const torch::Tensor tw_real,
    const torch::Tensor tw_imag,
    const torch::Tensor twinv_real,
    const torch::Tensor twinv_imag,
    int B,
    int H,
    int N,
    int N1
) {
    CHECK_INPUT(u_real);
    CHECK_INPUT(u_imag);
    CHECK_INPUT(kf_real);
    CHECK_INPUT(kf_imag);
    CHECK_INPUT(f_real);
    CHECK_INPUT(f_imag);
    CHECK_INPUT(finv_real);
    CHECK_INPUT(finv_imag);
    CHECK_INPUT(tw_real);
    CHECK_INPUT(tw_imag);
    CHECK_INPUT(twinv_real);
    CHECK_INPUT(twinv_imag);

    TORCH_CHECK(u_real.size(0) == B && u_real.size(1) == H 
        && u_real.size(2) == N1 && u_real.size(3) == N1, "u_real correct shape?");
    TORCH_CHECK(u_imag.size(0) == B && u_imag.size(1) == H 
        && u_imag.size(2) == N1 && u_imag.size(3) == N1, "u_imag correct shape?");
    TORCH_CHECK(kf_real.size(0) == H && kf_real.size(1) == N1 && kf_real.size(2) == N1, "kf_real correct shape?");
    TORCH_CHECK(kf_imag.size(0) == H && kf_imag.size(1) == N1 && kf_imag.size(2) == N1, "kf_imag correct shape?");
    TORCH_CHECK(f_real.size(0) == N1 && f_real.size(1) == N1, "f_real correct shape?");
    TORCH_CHECK(f_imag.size(0) == N1 && f_imag.size(1) == N1, "f_imag correct shape?");
    TORCH_CHECK(finv_real.size(0) == N1 && finv_real.size(1) == N1, "finv_real correct shape?");
    TORCH_CHECK(finv_imag.size(0) == N1 && finv_imag.size(1) == N1, "finv_imag correct shape?");
    TORCH_CHECK(tw_real.size(0) == N1 && tw_real.size(1) == N1, "tw_real correct shape?");
    TORCH_CHECK(tw_imag.size(0) == N1 && tw_imag.size(1) == N1, "tw_imag correct shape?");
    TORCH_CHECK(twinv_real.size(0) == N1 && twinv_real.size(1) == N1, "twinv_real correct shape?");
    TORCH_CHECK(twinv_imag.size(0) == N1 && twinv_imag.size(1) == N1, "twinv_imag correct shape?");

    torch::Tensor out = torch::empty({B, H, N1, N1}, u_real.options());

    c10::BFloat16 *u_real_ptr               = u_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *u_imag_ptr               = u_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_real_ptr              = kf_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_imag_ptr              = kf_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_real_ptr               = f_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_imag_ptr               = f_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_real_ptr            = finv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_imag_ptr            = finv_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_real_ptr              = tw_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_imag_ptr              = tw_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_real_ptr           = twinv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_imag_ptr           = twinv_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr                    = out.data_ptr<c10::BFloat16>();

    const bf16* u_real_bf                   = reinterpret_cast<const bf16*>(u_real_ptr);
    const bf16* u_imag_bf                   = reinterpret_cast<const bf16*>(u_imag_ptr);
    const bf16* kf_real_bf                  = reinterpret_cast<const bf16*>(kf_real_ptr);
    const bf16* kf_imag_bf                  = reinterpret_cast<const bf16*>(kf_imag_ptr);
    const bf16* f_real_bf                   = reinterpret_cast<const bf16*>(f_real_ptr);
    const bf16* f_imag_bf                   = reinterpret_cast<const bf16*>(f_imag_ptr);
    const bf16* finv_real_bf                = reinterpret_cast<const bf16*>(finv_real_ptr);
    const bf16* finv_imag_bf                = reinterpret_cast<const bf16*>(finv_imag_ptr);
    const bf16* tw_real_bf                  = reinterpret_cast<const bf16*>(tw_real_ptr);
    const bf16* tw_imag_bf                  = reinterpret_cast<const bf16*>(tw_imag_ptr);
    const bf16* twinv_real_bf               = reinterpret_cast<const bf16*>(twinv_real_ptr);
    const bf16* twinv_imag_bf               = reinterpret_cast<const bf16*>(twinv_imag_ptr);
          bf16* o_bf                        = reinterpret_cast<bf16*>(o_ptr);

    unsigned long mem_size = 1000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_fftconv_tk(
        u_real_bf, u_imag_bf, kf_real_bf, kf_imag_bf,
        f_real_bf, f_imag_bf, finv_real_bf, finv_imag_bf,
        tw_real_bf, tw_imag_bf, twinv_real_bf, twinv_imag_bf,
        o_bf,
        B, H, N, N1, mem_size
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);

    return out;
}
#else
#include "harness_async.impl"
#endif