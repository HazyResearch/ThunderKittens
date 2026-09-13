#include "kittens.cuh"
#include "pyutils/cuda_utils.cuh"
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/semaphore>
#include <cuda/pipeline>
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_FUSED_LAYERNORM
#endif

static constexpr int NUM_WORKERS = (2);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);

using namespace kittens;

template<kittens::ducks::rv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    #pragma unroll
    for ( int i = 0 ; i < dst.outer_dim ; i ++ ) { 
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            float rand = curand_uniform(&state);
            if (rand < keep_prob) {
                dst[i][j].x = base_types::constants<bf16>::zero();
                dst[i][j].y = base_types::constants<bf16>::zero();
            }
        }
    }
    mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<kittens::ducks::sv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    #pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        float rand = curand_uniform(&state);
        if (rand < keep_prob) {
            dst[cur] = base_types::constants<bf16>::zero();
        }
    }
    warp::mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<int _d_model> struct norm_globals {
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

template<int D>
__global__ __launch_bounds__(NUM_THREADS, 1)
void layernorm_tk(const __grid_constant__ norm_globals<D> g, int n_per_tile) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    int batch = blockIdx.y;
    int seq_start = blockIdx.x*2;

    extern __shared__ alignment_dummy __shm[];
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
        warp::load(norm_bias_s, g.norm_bias, {0,0,0,0});
        warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      

    warp::load_async(       x_s[warpid][tic], g.x,        {batch, 0, seq_start+warpid, 0});
    warp::load_async(residual_s[warpid][tic], g.residual, {batch, 0, seq_start+warpid, 0});
    __syncthreads();
    
    int n_blocks = g.n_per_tile/NUM_WORKERS; 
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        auto cur_idx  = (block + 0)*NUM_WORKERS + warpid;
        auto next_idx = (block + 1)*NUM_WORKERS + warpid; 

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            warp::load_async(       x_s[warpid][toc], g.x,        {batch, 0, seq_start+next_idx, 0});
            warp::load_async(residual_s[warpid][toc], g.residual, {batch, 0, seq_start+next_idx, 0});
        }
        load_async_wait();
        __syncwarp(); 

        dropout_mask(x_s[warpid][tic], g.dropout_p); 
        warp::add(residual_s[warpid][tic], residual_s[warpid][tic], x_s[warpid][tic]);         
        warp::store(g.o_resid, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
        __syncwarp();

        warp::sum(mean, residual_s[warpid][tic]);
        mean = mean / __float2bfloat16(d_model);
        warp::sub(residual_s[warpid][tic], residual_s[warpid][tic], mean);  
        warp::mul(x_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]);
        warp::sum(var, x_s[warpid][tic]);
        var = var / __float2bfloat16(d_model);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

        // compute norm
        warp::div(residual_s[warpid][tic], residual_s[warpid][tic], var);
        warp::mul(residual_s[warpid][tic], residual_s[warpid][tic], norm_weight_s); 
        warp::add(residual_s[warpid][tic], residual_s[warpid][tic], norm_bias_s);
        __syncwarp();

        // save output
        warp::store(g.o, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
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
    size_t B, 
    size_t N,
    cudaStream_t stream = nullptr
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
    kittens::detail::throw_if_cuda_error(
        cudaFuncSetAttribute(
            layernorm_tk<D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        ),
        "cudaFuncSetAttribute"
    );

    dim3 grid(n_tile_size, B, 1);
    layernorm_tk<D><<<grid,NUM_THREADS,mem_size,stream>>>(g, n_per_tile);
    kittens::detail::throw_if_cuda_error(cudaGetLastError(), "layernorm kernel launch");
}


#ifdef TK_COMPILE_FUSED_LAYERNORM
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>
#include <iostream>
std::tuple<at::Tensor, at::Tensor> fused_layernorm(
    const at::Tensor x, 
    const at::Tensor residual, 
    const at::Tensor norm_weight, 
    const at::Tensor norm_bias, 
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

    at::Tensor out = at::empty({b, n, d}, x.options());
    at::Tensor out_resid = at::empty({b, n, d}, x.options());

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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dispatch_layernorm(
        d_x_bf, d_residual_bf, 
        d_norm_weight_bf, d_norm_bias_bf, 
        d_o, d_o_resid, dropout_p,
        (size_t)b, (size_t)n, stream
    );

    return std::make_tuple(out, out_resid);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm", fused_layernorm, "LayerNorm TK. Takes tensors (x, residual, norm_weight, norm_bias, dropout_p). x, residual, norm_weight, norm_bias are bf16. dropout_p is float. Returns (B, H, N, 128) in bf16.");
}
#elif defined(TVM_FFI_COMPILE)
#include "pyutils/tvm_ffi_utils.cuh"

namespace {

// Caller-owned output buffers must not overlap each other or any input tensor.
// Violating this precondition results in undefined behavior.
void TkLayernorm(tvm::ffi::TensorView output,
                 tvm::ffi::TensorView output_residual,
                 tvm::ffi::TensorView input,
                 tvm::ffi::TensorView residual,
                 tvm::ffi::TensorView weight,
                 tvm::ffi::TensorView bias) {
    kittens::tvm_ffi::check_same_device(
        output, output_residual, input, residual, weight, bias
    );
    TVM_FFI_CHECK(output.ndim() == 3 && output_residual.ndim() == 3 &&
                      input.ndim() == 3 && residual.ndim() == 3,
                  ValueError)
        << "tk_layernorm expects output tensors and inputs to be 3-dimensional";
    TVM_FFI_CHECK(weight.ndim() == 1 && bias.ndim() == 1, ValueError)
        << "tk_layernorm expects weight and bias to be 1-dimensional";
    for (int axis = 0; axis < input.ndim(); ++axis) {
        TVM_FFI_CHECK(output.size(axis) == input.size(axis) &&
                          output_residual.size(axis) == input.size(axis) &&
                          residual.size(axis) == input.size(axis),
                      ValueError)
            << "output and residual shapes must match input at axis " << axis;
    }
    TVM_FFI_CHECK(input.size(2) == 1024, ValueError)
        << "tk_layernorm expects hidden dimension 1024";
    TVM_FFI_CHECK(weight.size(0) == 1024 && bias.size(0) == 1024, ValueError)
        << "tk_layernorm expects weight and bias length 1024";
    TVM_FFI_CHECK(input.size(1) % 16 == 0, ValueError)
        << "tk_layernorm expects sequence length to be a multiple of 16";

    bf16 *output_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(output);
    bf16 *output_residual_ptr =
        kittens::tvm_ffi::tensor_data_ptr<bf16>(output_residual);
    bf16 *input_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(input);
    bf16 *residual_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(residual);
    bf16 *weight_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(weight);
    bf16 *bias_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(bias);

    kittens::tvm_ffi::check_data_alignment(output, 16);
    kittens::tvm_ffi::check_data_alignment(output_residual, 16);
    kittens::tvm_ffi::check_data_alignment(input, 16);
    kittens::tvm_ffi::check_data_alignment(residual, 16);
    kittens::tvm_ffi::check_data_alignment(weight, 16);
    kittens::tvm_ffi::check_data_alignment(bias, 16);

    tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
    cudaStream_t stream = kittens::tvm_ffi::get_cuda_stream(input.device());
    dispatch_layernorm(
        input_ptr, residual_ptr,
        weight_ptr, bias_ptr,
        output_ptr, output_residual_ptr,
        0.0f, static_cast<size_t>(input.size(0)),
        static_cast<size_t>(input.size(1)), stream
    );
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tk_layernorm, TkLayernorm);
#else
#include "harness.impl"
#endif
