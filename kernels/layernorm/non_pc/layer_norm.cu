// #define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS (2) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

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
    mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<int _d_model, int _n_tile_size, int _n_per_tile> struct norm_globals {
    static constexpr int d_model = _d_model;
    static constexpr int n_tile_size = _n_tile_size;
    static constexpr int n_per_tile = _n_per_tile;
    static constexpr int dropout_p = 0.1;

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
};

template<int D, int N_TILE_SIZE, int N_PER_TILE>
__global__ __launch_bounds__(NUM_THREADS, 1)
void layernorm(const __grid_constant__ norm_globals<D, N_TILE_SIZE, N_PER_TILE> g) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    int batch     = blockIdx.x / g.n_tile_size;
    int seq_start = blockIdx.x % g.n_tile_size;

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
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.

    // global loads
    if (warpid == 0) { 
        load(norm_bias_s, g.norm_bias, {0,0,0,0});
        load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      

    load_async(       x_s[warpid][tic], g.x,        {batch, 0, seq_start+warpid, 0});
    load_async(residual_s[warpid][tic], g.residual, {batch, 0, seq_start+warpid, 0});
    __syncthreads();
    
    int n_blocks = g.n_per_tile/NUM_WORKERS; 
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  
        auto cur_idx  = (block + 0)*NUM_WORKERS + warpid;
        auto next_idx = (block + 1)*NUM_WORKERS + warpid; 

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            load_async(       x_s[warpid][toc], g.x,        {batch, 0, seq_start+next_idx, 0});
            load_async(residual_s[warpid][toc], g.residual, {batch, 0, seq_start+next_idx, 0});
        }

        dropout_mask(x_s[warpid][tic], g.dropout_p); 
        add(residual_s[warpid][tic], residual_s[warpid][tic], x_s[warpid][tic]);         
        store(g.o_resid, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
        __syncthreads();

        sum(mean, residual_s[warpid][tic]);
        mean = mean / __float2bfloat16(d_model);
        sub(residual_s[warpid][tic], residual_s[warpid][tic], mean);  
        mul(x_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]);
        sum(var, x_s[warpid][tic]);
        var = var / __float2bfloat16(d_model);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

        // compute norm
        div(residual_s[warpid][tic], residual_s[warpid][tic], var);
        mul(residual_s[warpid][tic], residual_s[warpid][tic], norm_weight_s); 
        add(residual_s[warpid][tic], residual_s[warpid][tic], norm_bias_s);

        // save output
        store(g.o, residual_s[warpid][tic], {batch, 0, seq_start+cur_idx, 0});
    }
}


#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void 
fused_ln_tk(
    float dropout_p, 
    const torch::Tensor x, 
    const torch::Tensor residual, 
    const torch::Tensor norm_weight, const torch::Tensor norm_bias, 
    torch::Tensor out,
    torch::Tensor out_resid
) {
    CHECK_INPUT(x);
    CHECK_INPUT(residual);
    CHECK_INPUT(out);
    CHECK_INPUT(out_resid);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(norm_bias);

    int batch = x.size(0);
    auto n    = x.size(1);

    TORCH_CHECK(batch == out.size(0) && batch == residual.size(0), "Differing batch sizes?");
    TORCH_CHECK(x.size(2) == d_model,           "x is d_model?");
    TORCH_CHECK(residual.size(2) == d_model,    "residual is d_model?");
    TORCH_CHECK(out.size(2) == d_model,         "out is d_model?");
    TORCH_CHECK(out_resid.size(2) == d_model,   "out_resid is d_model?");
    TORCH_CHECK(norm_weight.size(0) == d_model, "norm_weight is d_model?");
    TORCH_CHECK(norm_bias.size(0) == d_model,   "norm_bias is d_model?");

    TORCH_CHECK(x.size(1) % kittens::TILE_DIM == 0,        "sequence length is divisible by 16?");
    TORCH_CHECK(residual.size(1) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");
    TORCH_CHECK(out.size(1) % kittens::TILE_DIM == 0,      "sequence length is divisible by 16?");
    TORCH_CHECK(out_resid.size(1) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *x_ptr           = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *residual_ptr    = residual.data_ptr<c10::BFloat16>();
    c10::BFloat16 *norm_bias_ptr   = norm_bias.data_ptr<c10::BFloat16>();
    c10::BFloat16 *norm_weight_ptr = norm_weight.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr           = out.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_resid_ptr     = out_resid.data_ptr<c10::BFloat16>();

    const bf16* x_bf           = reinterpret_cast<const bf16*>(x_ptr);
    const bf16* residual_bf    = reinterpret_cast<const bf16*>(residual_ptr);
    const bf16* norm_bias_bf   = reinterpret_cast<const bf16*>(norm_bias_ptr);
    const bf16* norm_weight_bf = reinterpret_cast<const bf16*>(norm_weight_ptr);
          bf16* o_bf           = reinterpret_cast<bf16*>(o_ptr);
          bf16* o_resid_bf     = reinterpret_cast<bf16*>(o_resid_ptr);

    // launch variables
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    unsigned long mem_size = d_model*NUM_WORKERS*2*2*2 + d_model*2*2;
    cudaFuncSetAttribute(
        layernorm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    int n_c = 2;
    int n_h = n/n_c;
    

    layernorm<<<batch*n_h,threads,mem_size>>>(
        n_c, 
        dropout_p, 
        x_bf,
        residual_bf, 
        norm_weight_bf, norm_bias_bf, 
        o_bf, o_resid_bf
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
#else
#include "harness.impl"
#endif

