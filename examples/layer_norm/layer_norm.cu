#include "../../src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#define NUM_WORKERS (1) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D (64) // full model dimension

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
    int n, int has_residual, float dropout_p, 
    const bf16* __x, 
    bf16* __residual,
    bf16* __norm_weight, bf16* __norm_bias, 
    bf16* __mean, bf16* __var, bf16* __o
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *x_g             = reinterpret_cast<const bf16*>(__x)+blockIdx.x*(n*D);
    const bf16 *residual_g      = reinterpret_cast<const bf16*>(__residual)+blockIdx.x*(n*D);
    const bf16 *norm_weight_g   = reinterpret_cast<const bf16*>(__norm_weight)+blockIdx.x*(D);
    const bf16 *norm_bias_g     = reinterpret_cast<const bf16*>(__norm_bias)+blockIdx.x*(D);
          bf16 *mean_g          = reinterpret_cast<bf16*>(__mean)+blockIdx.x*(n);
          bf16 *var_g           = reinterpret_cast<bf16*>(__var)+blockIdx.x*(n);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)+blockIdx.x*(n*D);

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

    row_vec<tile_reg_1xD> norm_weight; // 4 registers
    row_vec<tile_reg_1xD> norm_bias;   // 4 registers
    load(norm_weight, norm_weight_s);
    load(norm_bias,   norm_bias_s  );

    int tic = 0, toc = 1;
    
    // iterate through the input
    int chunk_size = kittens::TILE_DIM;
    int n_blocks = n / chunk_size;
    for(int block = 0; block < n_blocks; block++) { 
        
        tile_reg_1xD x; // 4 registers
        tile_reg_1xD temp; // 4 registers
        tile_reg_1xD temp_squared; // 4 registers
        col_vec<tile_reg_1xD> mean; // 4 registers
        col_vec<tile_reg_1xD> var; // 4 registers
        tile_reg_1xD o; // 4 registers

        // hbm to smem
        int cur_idx = block*NUM_WORKERS + warpid;
        load(x_s, x_g + cur_idx * x_s.num_elements, D);

        // smem to reg
        load(x, x_s);

        // dropout on x
        if (dropout_p > 0) { dropout_mask(x, dropout_p); }

        // add residual
        if ( has_residual > 0 ) { 
            tile_reg_1xD residual; // 4 registers
            load(residual_s, residual_g + cur_idx * residual_s.num_elements, D);
            load(residual, residual_s);
            add(x, residual, x); 
        }

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
        store(mean_g + cur_idx * chunk_size, mean_s); // no stride when passing vectors
        store(var_s, var);
        store(var_g + cur_idx * chunk_size, var_s); // no stride when passing vectors
    }
}

#include "harness.impl"

