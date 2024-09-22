// #define TORCH_COMPILE
#include "include/kittens.cuh"

using namespace kittens;

#define NUM_WORKERS 1
#define NUM_THREADS NUM_WORKERS * kittens::WARP_THREADS

#define N_1024 1024

#define rt_cmplx_bf_32x32 rt_cmplx_bf<2, 2>
#define rt_cmplx_bf_32x32_col rt_cmplx_bf<2, 2, ducks::rt_layout::col>
#define rt_cmplx_fl_32x32 rt_cmplx_fl<2, 2>
#define st_cmplx_bf_32x32 st_cmplx_bf<2, 2>

#define st_cmplx_bf_32x1024 st_cmplx_bf<2, 64>


__device__ static void inner_loop(
    st_cmplx_bf_32x1024 u,
    const bf16 *kf_real, const bf16 *kf_imag, 
    st_cmplx_bf_32x32 fft_mat, st_cmplx_bf_32x32 fft_inv_mat, 
    const bf16 *tw_real, const bf16 *tw_imag, 
    const bf16 *twinv_real, const bf16 *twinv_imag, 
    bf16 *o, 
    int b, int h, int n, int n1,
    int B_TILE, int H_TILE
) {
    int warpid = kittens::warpid();

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;

    // Registers; everyone loads
    rt_cmplx_bf_32x32 a_reg;  
    rt_cmplx_fl_32x32 mma_reg;  
    rt_cmplx_bf_32x32 accum;       
    rt_cmplx_bf_32x32_col b_reg;

    // now se have 32x1024 and we want the strip of 1024
    int N_TILE = 32; 
    for (int h = 0; h < H_TILE; h++) {
        for (int b = 0; b < B_TILE; b++) {
            for (int i = 0; i < N_TILE; i ++) {
                // load block of u
                auto ref_real = kittens::subtile_inplace<2, 2>(u.real, 0, i);
                transpose_inplace(a_reg);

                // // x = x @ f_32_fft 
                // zero(mma_reg);
                // mma_AB(mma_reg, a_reg, fft_mat, mma_reg);
                // copy(a_reg, mma_reg);

                // x = x.transpose(-1, -2) # ... 16, 1K
                transpose_inplace(a_reg);

                // x = x * twiddle_factors_fft
                load(accum, tw_real, tw_imag, 32, 32);
                mul(a_reg, a_reg, accum);

                // // block = block @ f_32_fft
                // zero(mma_reg);
                // mma_AB(mma_reg, a_reg, fft_mat, mma_reg);
                // copy(a_reg, mma_reg);

                // store(o + u_start, a_reg.real, 1024);
            }
        }   
    }
}


__global__ void fftconv_tk(
    bf16 *u_real, bf16 *u_imag, 
    const bf16 *kf_real, const bf16 *kf_imag, 
    const bf16 *f_real, const bf16 *f_imag, 
    const bf16 *finv_real, const bf16 *finv_imag, 
    const bf16 *tw_32_1k_real, const bf16 *tw_32_1k_imag, 
    const bf16 *twinv_32_1k_real, const bf16 *twinv_32_1k_imag,
    const bf16 *tw_32_32_real, const bf16 *tw_32_32_imag, 
    const bf16 *twinv_32_32_real, const bf16 *twinv_32_32_imag,
    bf16 *o, 
    int b, int h, int n, int n1,
    int B_TILE, int H_TILE
){
    int warpid = kittens::warpid();

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;
    int offset = (b_start*b_stride) + (h_start*h_stride);

    // shared memory setup to load from hbm
    bf16 *u_real_g = reinterpret_cast<bf16*>(u_real) + offset; 
    bf16 *u_imag_g = reinterpret_cast<bf16*>(u_imag) + offset;
    const bf16 *f_real_g = reinterpret_cast<const bf16*>(f_real); 
    const bf16 *f_imag_g = reinterpret_cast<const bf16*>(f_imag); 
    const bf16 *finv_real_g = reinterpret_cast<const bf16*>(finv_real);
    const bf16 *finv_imag_g = reinterpret_cast<const bf16*>(finv_imag);

    // SMEM; 
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_cmplx_bf_32x1024 (&u)  = al.allocate<st_cmplx_bf_32x1024>(); 
    st_cmplx_bf_32x32 (&fft_mat) = al.allocate<st_cmplx_bf_32x32>();
    st_cmplx_bf_32x32 (&fft_inv_mat) = al.allocate<st_cmplx_bf_32x32>();
    if (warpid == 0) {
        load(u, u_real_g, u_imag_g, 1024, 1024);
        load(fft_mat, f_real_g, f_imag_g, n1, n1);
        load(fft_inv_mat, finv_real_g, finv_imag_g, n1, n1);
    }

    rt_cmplx_bf_32x32 a_reg;
    rt_cmplx_bf_32x32 b_reg;
    rt_cmplx_fl_32x32 mma_reg; 
    rt_cmplx_bf_32x32_col fft_mat_reg;
    load(fft_mat_reg, fft_mat);

    // compute the FFT
    int chunk_size = 32;
    int N_TILE = 32; // / chunk_size;
    for (int h = 0; h < H_TILE; h++) {
        for (int b = 0; b < B_TILE; b++) {
            for (int i = 0; i < N_TILE; i ++) {
                // load block of u
                int u_start = (
                    ((b_start + b) * b_stride) + 
                    ((h_start + h) * h_stride) + 
                    (i * chunk_size)
                ); 
                auto ref_real = kittens::subtile_inplace<2, 2>(u.real, 0, i);
                load(a_reg.real, ref_real);
                auto ref_imag = kittens::subtile_inplace<2, 2>(u.imag, 0, i);
                load(a_reg.imag, ref_imag);

                transpose_inplace(a_reg);

                // x = x @ f_32_fft    # ... 1K, 16
                // zero(mma_reg);
                // transpose_inplace(fft_mat_reg);
                // mma_AB(mma_reg, a_reg, fft_mat_reg, mma_reg);
                // copy(a_reg, mma_reg);

                // x = x.transpose(-1, -2) # ... 16, 1K
                transpose_inplace(a_reg);
                
                // x = x * twiddle_factors_fft 
                int tw_start = (i * chunk_size);
                load(b_reg, tw_32_1k_real + tw_start, tw_32_1k_imag + tw_start, 1024, 1024);
                mul(a_reg, a_reg, b_reg);

                // store the result back to u
                store(ref_real, a_reg.real);
                store(ref_imag, a_reg.imag);
            }
        }
    }

    // inner loop
    inner_loop(
        u, 
        kf_real, kf_imag, 
        fft_mat, fft_inv_mat, 
        tw_32_32_real, tw_32_32_imag, 
        twinv_32_32_real, twinv_32_32_imag, 
        o, 
        b, h, n, n1,
        B_TILE, H_TILE
    );

    // compute the IFFT (similar to FFT)
    load(fft_mat_reg, fft_inv_mat);
    for (int h = 0; h < H_TILE; h++) {
        for (int b = 0; b < B_TILE; b++) {
            for (int i = 0; i < N_TILE; i ++) {
                // load block of u
                int u_start = (
                    ((b_start + b) * b_stride) + 
                    ((h_start + h) * h_stride) + 
                    (i * chunk_size)
                );
                auto ref_real = kittens::subtile_inplace<2, 2>(u.real, 0, i);
                load(a_reg.real, ref_real);
                auto ref_imag = kittens::subtile_inplace<2, 2>(u.imag, 0, i);
                load(a_reg.imag, ref_imag);
                
                // x = x * twiddle_factors_ifft 
                int tw_start = (i * chunk_size);
                load(b_reg, twinv_32_1k_real + tw_start, twinv_32_1k_imag + tw_start, 1024, 1024);
                mul(a_reg, a_reg, b_reg);

                // x = x.transpose(-1, -2) 
                transpose_inplace(a_reg);

                // x = x @ f_32_ifft  
                // zero(mma_reg);
                // mma_AB(mma_reg, a_reg, fft_mat_reg, mma_reg);
                // copy(b_reg, mma_reg);

                // x = x.transpose(-1, -2)
                transpose_inplace(a_reg);

                // store the result back to u
                store(o + u_start, a_reg.real, 1024);
            }
        }
    }
}

void launch_fftconv_tk(
    bf16 *u_real, bf16 *u_imag,
    const bf16 *kf_real, const bf16 *kf_imag, 
    const bf16 *f_real, const bf16 *f_imag, 
    const bf16 *finv_real, const bf16 *finv_imag, 
    const bf16 *tw_32_1k_real, const bf16 *tw_32_1k_imag, 
    const bf16 *twinv_32_1k_real, const bf16 *twinv_32_1k_imag,
    const bf16 *tw_32_32_real, const bf16 *tw_32_32_imag, 
    const bf16 *twinv_32_32_real, const bf16 *twinv_32_32_imag,
    bf16 *o, 
    int b, int h, int n, int n1, const int B_TILE, const int H_TILE
) {

    // 1 warp for 32x32 case
    const dim3 block_dim{
        (unsigned int)(kittens::WARP_THREADS)
    };
    const dim3 grid_dim{
        (unsigned int)(b + B_TILE - 1) / B_TILE,
        (unsigned int)(h + H_TILE - 1) / H_TILE
    };

    long mem_size = 200000;

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

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(
        u_real, u_imag, 
        kf_real, kf_imag, 
        f_real, f_imag, 
        finv_real, finv_imag, 
        tw_32_1k_real, tw_32_1k_imag, 
        twinv_32_1k_real, twinv_32_1k_real,
        tw_32_32_real, tw_32_32_imag, 
        twinv_32_32_real, twinv_32_32_imag,
        o, 
        b, h, n, n1, B_TILE, H_TILE
    );
}

#include "harness_async.impl"
