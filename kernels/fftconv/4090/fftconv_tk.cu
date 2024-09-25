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


template<ducks::rt::all RT, ducks::st::all ST>
__device__  inline void subtile_load(RT &dst, ST &src, int row_id) {
    static_assert(RT::rows * RT::cols == ST::cols);
    for(int i = 0; i < RT::height; i++) {
        for(int j = 0; j < RT::width; j++) {
            int base_row = i*16 + laneid()/4;
            int base_col = j*16 + 2*(laneid()%4);
            int col = base_row*RT::cols + base_col;
            dst.tiles[i][j].data[0] = (*(bf16_2*)&src[{row_id, col}]);
            dst.tiles[i][j].data[1] = (*(bf16_2*)&src[{row_id, col + 8*RT::cols}]);
            dst.tiles[i][j].data[2] = (*(bf16_2*)&src[{row_id, col + 8}]);
            dst.tiles[i][j].data[3] = (*(bf16_2*)&src[{row_id, col + 8*RT::cols + 8}]);
        }
    }
}

template<ducks::rt::all RT, ducks::st::all ST>
__device__  inline void subtile_store(ST &dst, RT &src, int row_id) {
    static_assert(RT::rows * RT::cols == ST::cols);

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    
    for(int i = 0; i < RT::height; i++) {
        for(int j = 0; j < RT::width; j++) {
            int base_row = i*16 + laneid()/4;
            int base_col = j*16 + 2*(laneid()%4);
            int col = base_row*RT::cols + base_col;
            *(U2*)(&dst[{row_id, col}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            *(U2*)(&dst[{row_id, col + 8*RT::cols}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            *(U2*)(&dst[{row_id, col + 8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            *(U2*)(&dst[{row_id, col + 8*RT::cols + 8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}


__global__ void fftconv_tk(
    bf16 *u_real, bf16 *u_imag, 
    // const bf16 *kf_real, const bf16 *kf_imag, 

    const bf16 *f_real, const bf16 *f_imag, 
    // const bf16 *finv_real, const bf16 *finv_imag,
    
    // const bf16 *tw_32_1k_real, const bf16 *tw_32_1k_imag, 
    // const bf16 *twinv_32_1k_real, const bf16 *twinv_32_1k_imag,

    // const bf16 *tw_32_32_real, const bf16 *tw_32_32_imag, 
    // const bf16 *twinv_32_32_real, const bf16 *twinv_32_32_imag,

    bf16 *o, 

    int b, int h, int n, int n1,
    int B_TILE, int H_TILE
){

    printf("b: %d, h: %d, n: %d, n1: %d\n", b, h, n, n1);
    int warpid = kittens::warpid();
    int do_print = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;

    // shared memory allocation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_cmplx_bf_32x1024 (&u)  = al.allocate<st_cmplx_bf_32x1024>(); 
    st_cmplx_bf_32x32 (&fft_mat) = al.allocate<st_cmplx_bf_32x32>();
    st_cmplx_bf_32x32 (&fft_inv_mat) = al.allocate<st_cmplx_bf_32x32>();
    if (warpid == 0) {
        load(u, u_real, u_imag, 1024, 1024);
        load(fft_mat, f_real, f_imag, n1, n1);
        // load(fft_inv_mat, finv_real, finv_imag, n1, n1);
    }
    __syncthreads();

    // if (do_print) {
    //     int rows = u.real.rows;
    //     int cols = u.real.cols;
    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < 1; j ++ ) {
    //             printf("%f  ", __bfloat162float(u.real[{i, j}]));
    //         }
    //         printf("\n");
    //     }
    // }

    // register allocation
    rt_cmplx_bf_32x32 a_reg;
    rt_cmplx_bf_32x32 b_reg;
    rt_cmplx_fl_32x32 mma_reg; 
    rt_cmplx_bf_32x32 fft_mat_reg; // keep in register
    // rt_cmplx_bf_32x32_col fft_mat_inv_reg; // keep in register

    load(fft_mat_reg, fft_mat);
    // load(fft_mat_inv_reg, fft_inv_mat);

    // compute the FFT
    // int chunk_size = 1024;
    int N_TILE = 32;
    for (int h = 0; h < H_TILE; h++) {
        for (int b = 0; b < B_TILE; b++) {
            for (int i = 0; i < N_TILE; i ++) {
                // load block of u
                int u_start = (
                    // ((b_start + b) * b_stride) + 
                    // ((h_start + h) * h_stride) + 
                    (i * 32)
                ); 
                auto ref_real = kittens::subtile_inplace<2, 2>(
                    u.real, 0, i
                );
                auto ref_imag = kittens::subtile_inplace<2, 2>(u.imag, 0, i);
                load(a_reg.real, ref_real);
                load(a_reg.imag, ref_imag);

                // transpose
                // transpose_inplace(a_reg);

                rt_cmplx_bf_32x32_col a_reg_col;
                // swap_layout(a_reg_col, a_reg);

                // x = x @ f_32_fft 
                zero(mma_reg);
                // mma_AB(mma_reg, fft_mat_reg, a_reg_col,  mma_reg);
                transpose_inplace(fft_mat_reg);
                swap_layout(a_reg_col, fft_mat_reg);
                kittens::mma_AB(mma_reg, a_reg, a_reg_col,  mma_reg);
                copy(a_reg, mma_reg);

                // transpose
                // transpose_inplace(a_reg);
                // twiddle
                // int tw_start = (i * chunk_size);
                // load(b_reg, tw_32_1k_real + tw_start, tw_32_1k_imag + tw_start, 1024, 1024);
                // mul(a_reg, a_reg, b_reg);

                // store the result back to u
                store(ref_real, a_reg.real);
                store(ref_imag, a_reg.imag);
                store(o + u_start, a_reg.real, 1024);
                __syncthreads();
            }
        }
    }

    // if (do_print) {
    //     int rows = u.real.rows;
    //     int cols = u.real.cols;
    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < 3; j ++ ) {
    //             printf("%f  ", __bfloat162float(u.real[{i, j}]));
    //         }
    //         printf("\n");
    //     }
    // }

    // inner loop later...
    // int N_ROWS = 32; 
    // for (int h = 0; h < H_TILE; h++) {
    //     for (int b = 0; b < B_TILE; b++) {
    //         for (int i = 0; i < N_ROWS; i ++) {
    //             // u is 32x1024 and we want the vector of 1x1024
    //             subtile_load(a_reg.real, u.real, i);
    //             subtile_load(a_reg.imag, u.imag, i);
    //             // transpose    
    //             transpose_inplace(a_reg);
    //             // x = x @ f_32_fft  
    //             // zero(mma_reg);
    //             // mma_AB(mma_reg, a_reg, fft_mat_reg, mma_reg);
    //             // copy(a_reg, mma_reg);
    //             // transpose
    //             transpose_inplace(a_reg);
    //             // twiddle factors
    //             load(b_reg, tw_32_32_real, tw_32_32_imag, 32, 32);
    //             mul(a_reg, a_reg, b_reg);
    //             // x = x @ f_32_fft
    //             // zero(mma_reg);
    //             // mma_AB(mma_reg, a_reg, fft_mat_reg, mma_reg);
    //             // copy(a_reg, mma_reg);

    //             // pointwise multiplication with conv filter
    //             // int k_start = (((h_start + h) * h_stride) + (i * 32));
    //             // load(b_reg, kf_real + k_start, kf_imag + k_start, 32, 32);
    //             // mul(a_reg, a_reg, b_reg);

    //             // apply the ifft 
    //             // zero(mma_reg);
    //             // mma_AB(mma_reg, a_reg, fft_mat_inv_reg, mma_reg);
    //             // copy(a_reg, mma_reg);
    //             transpose_inplace(a_reg);
    //             // twiddle inverse
    //             load(b_reg, twinv_32_32_real, twinv_32_32_imag, 32, 32);
    //             mul(a_reg, a_reg, b_reg);
    //             // apply the ifft
    //             // zero(mma_reg);
    //             // mma_AB(mma_reg, a_reg, fft_mat_inv_reg, mma_reg);
    //             // copy(a_reg, mma_reg);
    //             // transpose
    //             transpose_inplace(a_reg);
    //             // write down the row
    //             subtile_store(u.real, a_reg.real, i);
    //             subtile_store(u.imag, a_reg.imag, i);
    //         }
    //     }   
    // }

    // // compute the IFFT (similar to FFT)
    // for (int h = 0; h < H_TILE; h++) {
    //     for (int b = 0; b < B_TILE; b++) {
    //         for (int i = 0; i < N_TILE; i ++) {
    //             // load block of u
    //             int u_start = (
    //                 ((b_start + b) * b_stride) + 
    //                 ((h_start + h) * h_stride) + 
    //                 (i * chunk_size)
    //             );
    //             auto ref_real = kittens::subtile_inplace<2, 2>(u.real, 0, i);
    //             load(a_reg.real, ref_real);
    //             auto ref_imag = kittens::subtile_inplace<2, 2>(u.imag, 0, i);
    //             load(a_reg.imag, ref_imag);
    //             // // x = x * twiddle_factors_ifft 
    //             // int tw_start = (i * chunk_size);
    //             // load(b_reg, twinv_32_1k_real + tw_start, twinv_32_1k_imag + tw_start, 1024, 1024);
    //             // mul(a_reg, a_reg, b_reg);
    //             // x = x.transpose(-1, -2) 
    //             transpose_inplace(a_reg);
    //             // x = x @ f_32_ifft  
    //             // zero(mma_reg);
    //             // mma_AB(mma_reg, a_reg, fft_mat_inv_reg, mma_reg);
    //             // copy(a_reg, mma_reg);
    //             // // x = x.transpose(-1, -2)
    //             transpose_inplace(a_reg);
    //             // store the result back to u
    //             store(o + u_start, a_reg.real, 1024);
    //         }
    //     }
    // }
}

void launch_fftconv_tk(
    bf16 *u_real, bf16 *u_imag,
    // const bf16 *kf_real, const bf16 *kf_imag, 

    const bf16 *f_real, const bf16 *f_imag, 
    // const bf16 *finv_real, const bf16 *finv_imag,

    // const bf16 *tw_32_1k_real, const bf16 *tw_32_1k_imag, 
    // const bf16 *twinv_32_1k_real, const bf16 *twinv_32_1k_imag,

    // const bf16 *tw_32_32_real, const bf16 *tw_32_32_imag, 
    // const bf16 *twinv_32_32_real, const bf16 *twinv_32_32_imag,

    bf16 *o, 

    int b, int h, int n, int n1, 
    const int B_TILE, const int H_TILE
) {

    // 1 warp for 32x32 case
    const dim3 block_dim{
        (unsigned int)(kittens::WARP_THREADS)
    };
    const dim3 grid_dim{
        (unsigned int)(b + B_TILE - 1) / B_TILE,
        (unsigned int)(h + H_TILE - 1) / H_TILE
    };

    // const dim3 grid_dim{
    //     (unsigned int)(1),
    //     (unsigned int)(1)
    // };

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
        // kf_real, kf_imag, 

        f_real, f_imag, 
        // finv_real, finv_imag, 

        // tw_32_1k_real, tw_32_1k_imag, 
        // twinv_32_1k_real, twinv_32_1k_real,

        // tw_32_32_real, tw_32_32_imag, 
        // twinv_32_32_real, twinv_32_32_imag,

        o, 
        
        b, h, n, n1, B_TILE, H_TILE
    );
}

#include "harness_async.impl"

