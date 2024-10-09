// #define TORCH_COMPILE
#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;

#define NUM_WORKERS 1
#define NUM_WARPS (NUM_WORKERS) // SA: make it type 4
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

// shared patterns
#define SQRT_N 32
#define rt_cmplx_bf_base crt_bf<SQRT_N, SQRT_N>
#define rt_cmplx_bf_base_col crt_bf<SQRT_N, SQRT_N, ducks::rt_layout::col>
#define rt_cmplx_fl_base crt_fl<SQRT_N, SQRT_N>
#define st_cmplx_bf_base cst_bf<SQRT_N, SQRT_N>

template<int b_tiles, int h_tiles, int h, int n, int n1>
struct fftconv_layout {
    using input_layout = gl<bf16, -1, h, n1, n1>;
    using filter_layout = gl<bf16, 1, h, n1, n1>;
    using fft_layout = gl<bf16, 1, 1, n1, n1>;

    using complex_input_layout = kittens::cgl<input_layout>;
    using complex_filter_layout = kittens::cgl<filter_layout>;
    using complex_fft_layout = kittens::cgl<fft_layout>;
    
    struct globals { 
        complex_input_layout u_g;
        input_layout o_real_g;
        complex_filter_layout kf_g;
        complex_fft_layout f_g, finv_g, tw_real_g, twinv_g;
    };
};
template<int _b_tiles, int _h_tiles, int _h, int _n, int _n1>
struct fftconv_template {
    static constexpr int b_tiles=_b_tiles, h_tiles=_h_tiles, h=_h, n=_n, n1=_n1;
    using layout = fftconv_layout<b_tiles, h_tiles, h, n, n1>;
};

template<typename T>
__global__ void fftconv_tk(typename T::layout::globals g) {
    int warpid = kittens::warpid();
    int H_TILE = T::h_tiles;
    int B_TILE = T::b_tiles;

    // Every block loads same seq tile
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;

    // Registers; everyone loads
    rt_cmplx_bf_base a_reg;       
    rt_cmplx_fl_base mma_reg;     
    rt_cmplx_bf_base accum;       
    rt_cmplx_bf_base_col b_reg;

    zero(a_reg);
    zero(mma_reg);
    zero(accum);
    zero(b_reg);

    // #pragma unroll
    for (int i = h_start; i < h_start+H_TILE; i++) {
        // #pragma unroll
        for (int j = b_start; j < b_start+B_TILE; j++) {            
            // X = F^T X
            load(a_reg, g.f_g, {0, 0, 0, 0});
            transpose_inplace(a_reg);
            load(b_reg, g.u_g, {j, i, 0, 0}); // needs to be imag too.
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, a_reg, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X * tw
            load(a_reg, g.tw_real_g, {0, 0, 0, 0});// needs to be imag too.
            kittens::mul(accum, accum, a_reg);

            // // X = XF
            load(b_reg, g.f_g, {0, 0, 0, 0}); // needs to be imag too.
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X * K_f^T
            load(a_reg, g.kf_g, {0, i, 0, 0});
            kittens::mul(accum, accum, a_reg);

            // X = XFinv
            load(b_reg, g.finv_g, {0, 0, 0, 0});
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X^T * twinv
            transpose_inplace(accum);
            load(a_reg, g.twinv_g, {0, 0, 0, 0});
            kittens::mul(accum, accum, a_reg);

            // Y = XFinv
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // Write Y^T to HBM
            transpose_inplace(accum);
            store(g.o_real_g, accum.real, {j, i, 0, 0});
        }
    }
}

template<typename T>
void launch_fftconv_tk(typename T::layout::globals g, int b, int h) {
    
    const int B_TILE = T::b_tiles; // Number of batches per SM
    const int H_TILE = T::h_tiles; // Number of height tiles per SM

    // 1 warp for 32x32 case
    const dim3 block_dim{
        (unsigned int)(NUM_THREADS)
    };
    const dim3 grid_dim{
        (unsigned int)(b + B_TILE - 1) / B_TILE,
        (unsigned int)(h + H_TILE - 1) / H_TILE
    };

    long mem_size = 1000;

    cudaFuncSetAttribute(
        fftconv_tk<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<T><<<grid_dim, block_dim, mem_size>>>(g);
}

#include "harness_async.impl"
