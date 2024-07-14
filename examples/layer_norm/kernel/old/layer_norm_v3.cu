// #define TORCH_COMPILE 

#include "src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS (2) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
const int d_model =  1024; 
const int d_model_tile = d_model / kittens::TILE_DIM;

using namespace kittens;
using layout = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_o = kittens::ducks::st_layout::swizzle;
using layout_reg = kittens::ducks::rt_layout::row;

#define vec_smem_1xD sv_bf<d_model_tile>
#define tile_smem_1xD st<bf16, 1, d_model_tile, layout>
#define vec_reg_1xD  row_vec<rt_bf<1, d_model_tile>>
#define col_vec_smem_1xD col_vec<rt_bf<1, d_model_tile>>
#define tile_reg_1xD  rt_bf<1, d_model_tile>


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

template<ducks::rv::all RV>
__device__ inline void compute_mean(float* mean, const RV residual) {
    using dtype = typename RV::dtype;
    using packed_type = typename base_types::packing<dtype>::unpacked_type;

    float sum = 0.0f;
    #pragma unroll
    for ( int i = 0 ; i < residual.outer_dim ; i ++ ) { 
        #pragma unroll
        for(int j = 0; j < residual.inner_dim; j++) {
            sum += __bfloat162float(residual[i][j].x);
            sum += __bfloat162float(residual[i][j].y);
        }
    }

    int leader_1; 
    if ( laneid() == 0 ) { leader_1 = 1; }     // put 1 --> 0
    else if ( laneid() == 2 ) { leader_1 = 3; }   // put 3 --> 2
    else { leader_1 = laneid(); } 
    float sum_neighbor = packed_shfl_sync(MASK_ALL, sum, leader_1);
    if ( laneid() == 0 ) { sum += sum_neighbor; }
    if ( laneid() == 2 ) { sum += sum_neighbor; }

    int leader_2; 
    if ( laneid() == 0 ) { leader_2 = 2; }    // put 2 --> 0 (0 should have full sum)
    else { leader_2 = laneid(); }
    float sum_neighbor_2 = packed_shfl_sync(MASK_ALL, sum, leader_2);
    if ( laneid() == 0 ) { sum += sum_neighbor_2; } 

    int leader_3 = 0;                         // everyone pull the value of thread 0
    float sum_neighbor_3 = packed_shfl_sync(MASK_ALL, sum, leader_3);
    sum = sum_neighbor_3; 

    *mean = sum / d_model;
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void fused_layer_norm(
    int n, int has_residual, 
    float dropout_p, 
    const bf16* __x,
    const bf16* __residual,
    const bf16* __norm_weight, 
    const bf16* __norm_bias, 
    bf16* __o,
    bf16* __o_residual
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *x_g             = reinterpret_cast<const bf16*>(__x)       +blockIdx.x*(n*d_model);
    const bf16 *residual_g      = reinterpret_cast<const bf16*>(__residual)+blockIdx.x*(n*d_model);
    const bf16 *norm_weight_g   = reinterpret_cast<const bf16*>(__norm_weight);
    const bf16 *norm_bias_g     = reinterpret_cast<const bf16*>(__norm_bias);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)             +blockIdx.x*(n*d_model);
          bf16 *o_residual_g    = reinterpret_cast<bf16*>(__o_residual)    +blockIdx.x*(n*d_model);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    vec_smem_1xD (&x_s)           [2][NUM_WORKERS] = al.allocate<vec_smem_1xD,2,NUM_WORKERS>();  // 128 bytes * 3
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
        load(norm_bias_s, norm_bias_g);
        load(norm_weight_s, norm_weight_g);
    }
    kittens::load_async(x_s[warpid][tic], x_g + warpid*d_model, d_model, barrier_cheat);
    if ( has_residual > 0 ) { 
        kittens::load_async(residual_s[warpid][tic], residual_g + warpid*d_model, d_model, barrier_cheat); 
    }

    // bf16 mean = 0.0f;
    float mean = 0.0f;
    float var = 0.0f;     
    vec_reg_1xD x, temp, temp_squared, residual;          // 4 registers each
    vec_reg_1xD norm_weight, norm_bias;  
    
    int n_blocks = n / NUM_WORKERS;
    for (int block = 0; block < n; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  

        // kick off load for the next block
        if(block < n_blocks - 1) {
            auto next_idx = (block + 1)*NUM_WORKERS + warpid; 
            kittens::load_async(x_s[warpid][toc], x_g + next_idx*d_model, d_model, barrier_cheat);
            kittens::load_async(residual_s[warpid][toc], residual_g + next_idx*d_model, d_model, barrier_cheat);
        }
        
        load(x, x_s[tic][warpid]);                            
        if (dropout_p > 0) { dropout_mask(x, dropout_p); }    // dropout on x
        if ( has_residual > 0 ) {                             // add residual
            load(residual, residual_s[tic][warpid]);
            add(residual, residual, x); 
        } else {
            copy(residual, x);
        }

        // sum(mean, residual_s[tic][warpid]);
        compute_mean(&mean, residual);                  
        mul(temp_squared, residual, residual);         // square
        compute_mean(&var, temp_squared);              // compute the variance
        var = sqrt(var + 1e-05f);                      // add norm.eps

        // compute norm
        load(norm_weight, norm_weight_s);
        load(norm_bias,   norm_bias_s);
        sub(temp, residual, __float2bfloat16(mean));   // center 
        div(temp, temp,     __float2bfloat16(var));
        mul(temp, temp, norm_weight); 
        add(temp, temp, norm_bias);

        // save output
        // store(o_residual_g + (block*NUM_WORKERS +warpid)*d_model, residual);
        store(o_g+ (block*NUM_WORKERS +warpid)*d_model, temp); 
    }
        
    // inspect stuff 
    // if ( blockIdx.x == 0 && threadIdx.x == 0 ) { 
    //     printf("mean: %f\n", mean); 
    //     printf("var: %f\n", var); 
    //     printf("");
    // }
}


#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void 
fused_ln_tk(
    int has_residual, 
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
    unsigned long mem_size = 10260;
    cudaFuncSetAttribute(
        fused_layer_norm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fused_layer_norm<<<batch*n,threads,mem_size>>>(
        n, has_residual, 
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
