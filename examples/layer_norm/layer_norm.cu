#include "../../src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (1) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D (64) // full model dimension


using namespace kittens;
using layout = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_o = kittens::ducks::st_layout::swizzle;


__global__ __launch_bounds__(NUM_THREADS, 1)
void fused_layer_norm(
    int n, const bf16* __x, 
    bf16* __norm_weight, bf16* __norm_bias, 
    bf16* __mean, bf16* __var, bf16* __o
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *x_g             = reinterpret_cast<const bf16*>(__x)+blockIdx.x*(n*D);
    const bf16 *norm_weight_g   = reinterpret_cast<const bf16*>(__norm_weight)+blockIdx.x*(D);
    const bf16 *norm_bias_g     = reinterpret_cast<const bf16*>(__norm_bias)+blockIdx.x*(D);
          bf16 *mean_g          = reinterpret_cast<bf16*>(__mean)+blockIdx.x*(n);
          bf16 *var_g           = reinterpret_cast<bf16*>(__var)+blockIdx.x*(n);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)+blockIdx.x*(n*D);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf_1x4<layout> (&x_s)  = al.allocate<st_bf_1x4<layout>>(); 
    st_bf_1x4<layout> (&o_s)  = al.allocate<st_bf_1x4<layout>>();
    col_vec<st_bf_1x4<layout>> (&mean_s)  = al.allocate<col_vec<st_bf_1x4<layout>>>();
    col_vec<st_bf_1x4<layout>> (&var_s)  = al.allocate<col_vec<st_bf_1x4<layout>>>();

    // norms: load in the full thing.
    row_vec<st_bf_1x4<layout>> (&norm_weight_s)  = al.allocate<row_vec<st_bf_1x4<layout>>>(); 
    row_vec<st_bf_1x4<layout>> (&norm_bias_s  )  = al.allocate<row_vec<st_bf_1x4<layout>>>(); 
    load(norm_weight_s, norm_weight_g);
    load(norm_bias_s  , norm_bias_g);
    row_vec<rt_bf_1x4<>> norm_weight; // 4 registers
    row_vec<rt_bf_1x4<>> norm_bias;   // 4 registers
    load(norm_weight, norm_weight_s);
    load(norm_bias, norm_bias_s    );

    int tic = 0, toc = 1;
    
    // iterate through the input
    int n_blocks = n / (kittens::TILE_DIM);
    for(int block = 0; block < n_blocks; block++) { 
        
        rt_bf_1x4<> x; // 4 registers
        rt_bf_1x4<> temp; // 4 registers
        rt_bf_1x4<> temp_squared; // 4 registers
        col_vec<rt_bf_1x4<>> mean; // 4 registers
        col_vec<rt_bf_1x4<>> var; // 4 registers
        rt_bf_1x4<> o; // 4 registers

        // hbm to smem
        int cur_idx = block*NUM_WORKERS + warpid;
        load(x_s, x_g + cur_idx * x_s.num_elements, D);

        // smem to reg
        load(x, x_s);

        // compute the mean
        zero(mean);
        row_sum(mean, x, mean);
        div(mean, mean, __float2bfloat16(D));

        // compute the variance
        zero(var);
        sub_row(temp, x, mean);  // center 
        mul(temp_squared, temp, temp);   // square
        row_sum(var, temp_squared, var);
        div(var, var, __float2bfloat16(D));
        add(var, var, __float2bfloat16(1e-05f));
        sqrt(var, var);

        // TODO: write function
        div_row(temp, temp, var);
        mul_col(temp, temp, norm_weight);
        add_col(temp, temp, norm_bias);

        // copy
        copy(o, temp);

        // store reg to smem
        store(o_s, o);

        // store smem to hbm
        store(o_g + cur_idx * o_s.num_elements, o_s, D);
        
        // inspect stuff 
        store(mean_s, mean);
        store(mean_g + cur_idx * 16, mean_s); // no stride when passing vectors
        store(var_s, var);
        store(var_g + cur_idx * 16, var_s); // no stride when passing vectors
    }
}

#include "harness.impl"

