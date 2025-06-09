// #define TORCH_COMPILE
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include "prototype.dp.hpp"

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

    /*
    DPCT1128:425: The type "typename fftconv_template<8, 8, 16, 1024,
    32>::layout::globals" is not device copyable for copy constructor, move
    constructor, non trivially copyable field "u_g", non trivially copyable
    field "o_real_g", non trivially copyable field "kf_g", non trivially
    copyable field "f_g", non trivially copyable field "finv_g", non trivially
    copyable field "tw_real_g" and non trivially copyable field "twinv_g"
    breaking the device copyable requirement. It is used in the SYCL kernel,
    please rewrite the code.
    */
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

template <typename T>
/*
DPCT1110:420: The total declared local variable size in device function
fftconv_tk exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void fftconv_tk(typename T::layout::globals g) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int warpid = kittens::warpid();
    int H_TILE = T::h_tiles;
    int B_TILE = T::b_tiles;

    // Every block loads same seq tile
    int h_start = item_ct1.get_group(1) * H_TILE;
    int b_start = item_ct1.get_group(2) * B_TILE;

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
    const dpct::dim3 block_dim{(unsigned int)(NUM_THREADS)};
    const dpct::dim3 grid_dim{(unsigned int)(b + B_TILE - 1) / B_TILE,
                              (unsigned int)(h + H_TILE - 1) / H_TILE};

    long mem_size = 1000;

    /*
    DPCT1026:603: The call to cudaFuncSetAttribute was removed because SYCL
    currently does not support corresponding setting.
    */

    /*
    DPCT1129:426: The type "typename fftconv_template<8, 8, 16, 1024,
    32>::layout::globals" is used in the SYCL kernel, but it is not device
    copyable. The sycl::is_device_copyable specialization has been added for
    this type. Please review the code.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            cgh.depends_on(
                dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                             exp_props,
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     fftconv_tk<T>(g);
                                 });
        });
    }
}

#include "harness_async.impl"
