#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include <dpct/rng_utils.hpp>

#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_FUSED_LAYERNORM
#endif

#define NUM_WORKERS (2) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using namespace kittens;

template<kittens::ducks::rv::all T>
void dropout_mask(T &dst, float keep_prob) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long long seed = 0;
    const int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>
        state;
    state = dpct::rng::device::rng_generator<
        oneapi::mkl::rng::device::philox4x32x10<1>>(
        seed, {0, static_cast<std::uint64_t>(idx * 4)});

#pragma unroll
    for ( int i = 0 ; i < dst.outer_dim ; i ++ ) { 
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            float rand =
                state.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
            if (rand < keep_prob) {
                dst[i][j].x = base_types::constants<bf16>::zero();
                dst[i][j].y = base_types::constants<bf16>::zero();
            }
        }
    }
    mul(dst, dst, sycl::ext::intel::math::float2bfloat16(1 / (1 - keep_prob)));
}

template<kittens::ducks::sv::all T>
void dropout_mask(T &dst, float keep_prob) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long long seed = 0;
    const int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>
        state;
    state = dpct::rng::device::rng_generator<
        oneapi::mkl::rng::device::philox4x32x10<1>>(
        seed, {0, static_cast<std::uint64_t>(idx * 4)});

#pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        float rand =
            state.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
        if (rand < keep_prob) {
            dst[cur] = base_types::constants<bf16>::zero();
        }
    }
    mul(dst, dst, sycl::ext::intel::math::float2bfloat16(1 / (1 - keep_prob)));
}

/*
DPCT1128:375: The type "norm_globals<1024>" is not device copyable for copy
constructor, move constructor, non trivially copyable field "x", non trivially
copyable field "residual", non trivially copyable field "o", non trivially
copyable field "o_resid", non trivially copyable field "norm_weight" and non
trivially copyable field "norm_bias" breaking the device copyable requirement.
It is used in the SYCL kernel, please rewrite the code.
*/
template <int _d_model> struct norm_globals {
    static constexpr int d_model = _d_model;
    static constexpr int dropout_p = 0.0;

    // types
    using vec_smem_1xD  = sv_bf<d_model>;
    using tile_smem_1xD = st<bf16, 1, d_model>;
    using tile_reg_1xD  = rt_bf<1, d_model>;

    // global descriptors
    using x_gl            = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl     = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_gl            = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_resid_gl      = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl  = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_bias_gl    = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    // global pointers
    x_gl x;
    residual_gl residual;
    o_gl o;
    o_resid_gl o_resid;
    norm_weight_gl norm_weight;
    norm_bias_gl norm_bias;

    const int n_tile_size;
    const int n_per_tile;
};
template <int _d_model>
struct sycl::is_device_copyable<norm_globals<_d_model>> : std::true_type {};

template<int D>

void layernorm_tk(const norm_globals<D> g, int n_per_tile,
                  uint8_t *dpct_local) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    int batch = item_ct1.get_group(1);
    int seq_start = item_ct1.get_group(2) * 2;

    auto __shm = (alignment_dummy *)dpct_local;
    shared_allocator al((int*)&__shm[0]);

    static constexpr int d_model = D;
    using vec_smem_1xD = sv_bf<d_model>;
    using tile_smem_1xD = st<bf16, 1, d_model>;
    using tile_reg_1xD = rt_bf<1, d_model>;

    vec_smem_1xD (&x_s)           [2][NUM_WORKERS] = al.allocate<vec_smem_1xD,2,NUM_WORKERS>();
    vec_smem_1xD (&residual_s)    [2][NUM_WORKERS] = al.allocate<vec_smem_1xD,2,NUM_WORKERS>();  
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>(); 
    vec_smem_1xD (&norm_bias_s  ) = al.allocate<vec_smem_1xD>();                  

    // pipelining
    int tic = 0, toc = 1;

    // global loads
    if (warpid == 0) { 
        load(norm_bias_s, g.norm_bias, {0,0,0,0});
        load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }

    bf16 mean = sycl::ext::intel::math::float2bfloat16(0.0f);
    bf16 var = sycl::ext::intel::math::float2bfloat16(0.0f);

    load_async(       x_s[warpid][tic], g.x,        {batch, 0, seq_start+warpid, 0});
    load_async(residual_s[warpid][tic], g.residual, {batch, 0, seq_start+warpid, 0});
    /*
    DPCT1065:549: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int n_blocks = g.n_per_tile/NUM_WORKERS; 
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        auto cur_idx  = (block + 0)*NUM_WORKERS + warpid;
        auto next_idx = (block + 1)*NUM_WORKERS + warpid; 

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            load_async(       x_s[warpid][toc], g.x,        {batch, 0, seq_start+next_idx, 0});
            load_async(residual_s[warpid][toc], g.residual, {batch, 0, seq_start+next_idx, 0});
        }
        load_async_wait();
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

        dropout_mask(x_s[warpid][tic], g.dropout_p); 
        add(residual_s[warpid][tic], residual_s[warpid][tic], x_s[warpid][tic]);         
        store(g.o_resid, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

        sum(mean, residual_s[warpid][tic]);
        mean = mean / sycl::ext::intel::math::float2bfloat16(d_model);
        sub(residual_s[warpid][tic], residual_s[warpid][tic], mean);  
        mul(x_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]);
        sum(var, x_s[warpid][tic]);
        var = var / sycl::ext::intel::math::float2bfloat16(d_model);
        var = sycl::ext::intel::math::float2bfloat16(
            sycl::sqrt(sycl::ext::intel::math::bfloat162float(
                var + sycl::ext::intel::math::float2bfloat16(1e-05f))));

        // compute norm
        div(residual_s[warpid][tic], residual_s[warpid][tic], var);
        mul(residual_s[warpid][tic], residual_s[warpid][tic], norm_weight_s); 
        add(residual_s[warpid][tic], residual_s[warpid][tic], norm_bias_s);
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

        // save output
        store(g.o, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
    }
}

void dispatch_layernorm(
    bf16 *d_x_bf,
    bf16 *d_residual_bf,
    bf16 *d_norm_weight_bf,
    bf16 *d_norm_bias_bf,
    bf16 *d_o,
    bf16 *d_o_resid,
    float dropout_p,
    int B, int N
) {
    constexpr size_t D = 1024;

    // types
    // error: invalid narrowing conversion from "int" to "size_t"
    using vec_smem_1xD  = sv_bf<static_cast<size_t>(D)>;
    using tile_smem_1xD = st<bf16, 1, static_cast<size_t>(D)>;
    

    // global descriptors
    using x_gl           = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl    = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_gl           = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using o_resid_gl     = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_bias_gl   = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    // global pointers
    using globals = norm_globals<D>;
    x_gl  x_arg{d_x_bf, B, 1, N, D};
    residual_gl  residual_arg{d_residual_bf, B, 1, N, D};
    o_gl  o_arg{d_o, B, 1, N, D};
    o_resid_gl  o_resid_arg{d_o_resid, B, 1, N, D};
    norm_weight_gl norm_weight_arg{d_norm_weight_bf, 1, 1, 1, D};
    norm_bias_gl norm_bias_arg{d_norm_bias_bf, 1, 1, 1, D};

    const int n_tile_size = N / 2;
    const int n_per_tile = 2;
    globals g{x_arg, residual_arg, o_arg, o_resid_arg, norm_weight_arg, norm_bias_arg,
    n_tile_size, n_per_tile};

    unsigned long mem_size = 25480;
    /*
    DPCT1026:550: The call to cudaFuncSetAttribute was removed because SYCL
    currently does not support corresponding setting.
    */

    dpct::dim3 grid(n_tile_size, B, 1);
    /*
    DPCT1129:376: The type "norm_globals<1024>" is used in the SYCL kernel, but
    it is not device copyable. The sycl::is_device_copyable specialization has
    been added for this type. Please review the code.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(mem_size), cgh);

            cgh.depends_on(
                dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, NUM_THREADS),
                                  sycl::range<3>(1, 1, NUM_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        layernorm_tk<D>(
                            g, n_per_tile,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    dpct::get_current_device().queues_wait_and_throw();
}


#ifdef TK_COMPILE_FUSED_LAYERNORM
#include "pyutils/torch_helpers.cuh"
#include <iostream>
std::tuple<torch::Tensor, torch::Tensor> fused_layernorm(
    const torch::Tensor x, 
    const torch::Tensor residual, 
    const torch::Tensor norm_weight, 
    const torch::Tensor norm_bias, 
    float dropout_p
) {
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(norm_bias);

    int b = x.size(0);
    int n = x.size(1);
    constexpr int d = 1024; // hard coded for this kernel

    TORCH_CHECK(b == residual.size(0), "Differing b sizes?");
    TORCH_CHECK(x.size(2) == d, "x is d_model?");
    TORCH_CHECK(residual.size(2) == d, "residual is d_model?");
    TORCH_CHECK(norm_weight.size(0) == d, "norm_weight is d_model?");
    TORCH_CHECK(norm_bias.size(0) == d, "norm_bias is d_model?");

    TORCH_CHECK(x.size(1) % kittens::TILE_ROW_DIM<bf16> == 0,        "sequence length is divisible by 16?");
    TORCH_CHECK(residual.size(1) % kittens::TILE_ROW_DIM<bf16> == 0, "sequence length is divisible by 16?");

    torch::Tensor out = torch::empty({b, n, d}, x.options());
    torch::Tensor out_resid = torch::empty({b, n, d}, x.options());

    // convert to bf16
    c10::BFloat16 *x_ptr           = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *residual_ptr    = residual.data_ptr<c10::BFloat16>();
    c10::BFloat16 *norm_bias_ptr   = norm_bias.data_ptr<c10::BFloat16>();
    c10::BFloat16 *norm_weight_ptr = norm_weight.data_ptr<c10::BFloat16>();
    
    bf16* d_x_bf = reinterpret_cast<bf16*>(x_ptr);
    bf16* d_residual_bf = reinterpret_cast<bf16*>(residual_ptr);
    bf16* d_norm_bias_bf = reinterpret_cast<bf16*>(norm_bias_ptr);
    bf16* d_norm_weight_bf = reinterpret_cast<bf16*>(norm_weight_ptr);
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    bf16 *d_o_resid = reinterpret_cast<bf16*>(out_resid.data_ptr<c10::BFloat16>());

    dispatch_layernorm(
        d_x_bf, d_residual_bf, 
        d_norm_weight_bf, d_norm_bias_bf, 
        d_o, d_o_resid, dropout_p,
        b, n
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    return std::make_tuple(out, out_resid);
}
#else
#include "harness.impl"
#include <sycl/ext/intel/math.hpp>

#endif

