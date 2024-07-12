#define TORCH_COMPILE 

#include "src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS (1) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
const int d_model =  64; 

using namespace kittens;
using layout = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_o = kittens::ducks::st_layout::swizzle;

#define tile_smem_1xD st_bf_1x4<layout>
#define tile_reg_1xD rt_bf_1x4<>

template<kittens::ducks::rt::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                float rand = curand_uniform(&state);
                if (rand < keep_prob) {
                    dst.tiles[i][j].data[k] = base_types::constants<bf16_2>::zero();
                }
            }
        }
    }
    mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}


__global__ __launch_bounds__(NUM_THREADS, 1)
void fused_layer_norm(
    int n, int has_residual, 
    float dropout_p, 
    const bf16* __x,
    const bf16* __residual,
    const bf16* __norm_weight, const bf16* __norm_bias, 
    bf16* __o,
    bf16* __o_residual
    // ,
    // bf16* __mean, bf16* __var
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *x_g             = reinterpret_cast<const bf16*>(__x)+blockIdx.x*(n*d_model);
    const bf16 *residual_g      = reinterpret_cast<const bf16*>(__residual)+blockIdx.x*(n*d_model);
    const bf16 *norm_weight_g   = reinterpret_cast<const bf16*>(__norm_weight)+blockIdx.x*(d_model);
    const bf16 *norm_bias_g     = reinterpret_cast<const bf16*>(__norm_bias)+blockIdx.x*(d_model);
        //   bf16 *mean_g          = reinterpret_cast<      bf16*>(__mean)+blockIdx.x*(n);
        //   bf16 *var_g           = reinterpret_cast<      bf16*>(__var)+blockIdx.x*(n);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)+blockIdx.x*(n*d_model);
          bf16 *o_residual_g    = reinterpret_cast<bf16*>(__o_residual)+blockIdx.x*(n*d_model);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_smem_1xD (&x_s)  = al.allocate<tile_smem_1xD>();
    tile_smem_1xD (&residual_s)  = al.allocate<tile_smem_1xD>();  
    tile_smem_1xD (&o_s)  = al.allocate<tile_smem_1xD>();
    col_vec<tile_smem_1xD> (&mean_s)  = al.allocate<col_vec<tile_smem_1xD>>();
    col_vec<tile_smem_1xD> (&var_s)  = al.allocate<col_vec<tile_smem_1xD>>();

    // norms: load in the full thing.
    row_vec<tile_smem_1xD> (&norm_weight_s)  = al.allocate<row_vec<tile_smem_1xD>>(); 
    row_vec<tile_smem_1xD> (&norm_bias_s  )  = al.allocate<row_vec<tile_smem_1xD>>(); 
    load(norm_weight_s, norm_weight_g);
    load(norm_bias_s  , norm_bias_g);

    row_vec<tile_reg_1xD> norm_weight; 
    row_vec<tile_reg_1xD> norm_bias;  
    load(norm_weight, norm_weight_s);
    load(norm_bias,   norm_bias_s  );

    int tic = 0, toc = 1;
    // float dropout_p = 0.1;

    // kittens::load_async(q[tic][warpid], _q + warpid*qk_tile_elements, d,  qkv_barrier);
    // auto block = cooperative_groups::this_thread_block();
    // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> x_barrier;
    // if (threadIdx.x == 0) {init(&x_barrier, block.size());}
    // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    // if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    // block.sync(); // Make sure no gets to the barrier before its initialized.
    // if (warpid == 1) {
    //     kittens::load_async(x_s[tic], x_g, D, x_barrier); 
    //     kittens::load_async(x_s[tic], x_g, D, x_barrier); 
        
    
    // iterate through the input
    int chunk_size = kittens::TILE_DIM;
    int n_blocks = n / chunk_size;
    for(int block = 0; block < n_blocks; block++, tic ^= 1, toc ^= 1) { 
        // x_barrier.arrive_and_wait();
        
        tile_reg_1xD x;            // 4 registers
        tile_reg_1xD temp;         // 4 registers
        tile_reg_1xD temp_squared; // 4 registers
        col_vec<tile_reg_1xD> mean;
        col_vec<tile_reg_1xD> var; 
        tile_reg_1xD o; // 4 registers
        
        int cur_idx = block*NUM_WORKERS + warpid;
        load(x_s, x_g + cur_idx * x_s.num_elements, d_model); // hbm to smem
        load(x, x_s); // smem to reg

        if (dropout_p > 0) { dropout_mask(x, dropout_p); } // dropout on x

        // add residual
        tile_reg_1xD residual; // 4 registers
        if ( has_residual > 0 ) { 
            load(residual_s, residual_g + cur_idx * residual_s.num_elements, d_model);
            load(residual, residual_s);
            add(residual, residual, x); 
        } else {
            copy(residual, x);
        }

        // compute the mean
        zero(mean);
        row_sum(mean, residual, mean);
        div(mean, mean, __float2bfloat16(d_model));

        // compute the variance
        zero(var);
        sub_row(temp, residual, mean);   // center 
        mul(temp_squared, temp, temp);   // square
        row_sum(var, temp_squared, var);
        div(var, var, __float2bfloat16(d_model));
        add(var, var, __float2bfloat16(1e-05f)); // add norm.eps
        sqrt(var, var);

        div_row(temp, temp, var);
        mul_col(temp, temp, norm_weight);
        add_col(temp, temp, norm_bias);

        // save output
        copy(o, temp);
        store(o_s, temp); // store reg to smem
        store(o_g + cur_idx * o_s.num_elements, o_s, d_model); // store smem to hbm

        // save residual
        store(residual_s, residual); // store reg to smem
        store(o_residual_g + cur_idx * residual_s.num_elements, residual_s, d_model); // store smem to hbm
        
        // inspect stuff 
        // store(mean_s, mean);
        // store(mean_g + cur_idx * chunk_size, mean_s); // no stride when passing vectors
        // store(var_s, var);
        // store(var_g + cur_idx * chunk_size, var_s); // no stride when passing vectors
    }
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
    auto n       = x.size(1);

    unsigned long mem_size = 100000;
    cudaFuncSetAttribute(
        fused_layer_norm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fused_layer_norm<<<batch,threads,mem_size>>>(
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
