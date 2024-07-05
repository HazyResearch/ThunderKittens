#include "src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <iostream>

#define NUM_WORKERS (4) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;

template<ducks::st::all ST>
__device__ inline void cumulative_add(ST &dst, const ST &inc) {
    // first do a reduction for each col
    constexpr int num_elts = (ST::cols + WARP_THREADS - 1) / WARP_THREADS;
    float acc[num_elts];  
    for (auto i = 0; i < num_elts; i++) {
        auto col = (kittens::laneid() + (i * WARP_THREADS)); //0..31
        if (col < dst.cols) {
            acc[i] = __bfloat162float(dst[{dst.rows-1,col}]); // acc equal to the last row of dst
            for (auto row = 0; row < dst.rows; row++) {
                acc[i] += __bfloat162float(inc[{row,col}]);
                dst[{row,col}] = __float2bfloat16(acc[i]);
            }
        }
        __syncwarp();
    }
}

template<ducks::rt::all RT>
__device__ inline void cumulative_add(RT &dst, const RT &src, const int block_idx) {
    using dtype = typename RT::dtype;
    using packed_type = typename base_types::packing<dtype>::unpacked_type;
    // we know that src and dst will have the same dtype and shape by definition

    if ( threadIdx.x == 0 && blockIdx.x == 0 && block_idx == 0 ) { 
        printf("Inside the cumulative_add() function.\n"); 
    }
    const int row = laneid() / 4;
    __syncthreads();

    // step 0. extract the last row of dst (tiles dst 1 to src 0 and dst 2 to src 3). 
    int leader_step_0; 
    if ( row % 8 < 7 ) { leader_step_0 = laneid() + 4*(7 - row);  }
    else { leader_step_0 = laneid(); }
    __syncthreads();
    row_vec<rt_bf<1,1>> last_row_1, last_row_3;
    rt_bf<1,1> broadcast_last_row_1, broadcast_last_row_3;
    dtype copy_accum_packed_1 = dst.tiles[0][0].data[1];
    dtype copy_accum_packed_3 = dst.tiles[0][0].data[3];
    __syncthreads();

    copy_accum_packed_1 = packed_shfl_sync(MASK_ALL, copy_accum_packed_1, leader_step_0);
    last_row_1[0][0] = copy_accum_packed_1;
    broadcast_col(broadcast_last_row_1, last_row_1);
    dtype (*broadcast_last_row_1_) = reinterpret_cast<dtype*>(&broadcast_last_row_1);
    __syncthreads();

    copy_accum_packed_3 = packed_shfl_sync(MASK_ALL, copy_accum_packed_3, leader_step_0);
    last_row_3[0][0] = copy_accum_packed_3;
    broadcast_col(broadcast_last_row_3, last_row_3);
    dtype (*broadcast_last_row_3_) = reinterpret_cast<dtype*>(&broadcast_last_row_3);
    __syncthreads();

    // step 1. even row i ships its values to odd row i+1
    const int leader = (row % 2 == 1) ? laneid() - 4: laneid();
    dtype pull_0 = packed_shfl_sync(MASK_ALL, src.tiles[0][0].data[0], leader);
    dtype pull_1 = packed_shfl_sync(MASK_ALL, src.tiles[0][0].data[1], leader);
    dtype pull_2 = packed_shfl_sync(MASK_ALL, src.tiles[0][0].data[2], leader);
    dtype pull_3 = packed_shfl_sync(MASK_ALL, src.tiles[0][0].data[3], leader);
    dtype accum_packed_0 = src.tiles[0][0].data[0];
    dtype accum_packed_1 = src.tiles[0][0].data[1];
    dtype accum_packed_2 = src.tiles[0][0].data[2];
    dtype accum_packed_3 = src.tiles[0][0].data[3];
    __syncthreads();
    if ( row % 2 == 1) {  accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0); }
    if ( row % 2 == 1) {  accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2); }
    if ( row % 2 == 1) {  accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1); }
    if ( row % 2 == 1) {  accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3); }
    __syncthreads();

    // step 2. each row pulls from the last row of the prior two rows cumsum
    int leader_step_2;
    if (row % 4 == 2 || row % 4 == 3 ) { 
        leader_step_2 = (row % 2 == 0) ? laneid() - 4 : laneid() - 8;
    } else {  
        leader_step_2 = laneid();  
    }  
    __syncthreads();
    dtype pull_0_ = packed_shfl_sync(MASK_ALL, accum_packed_0, leader_step_2);
    dtype pull_2_ = packed_shfl_sync(MASK_ALL, accum_packed_2, leader_step_2);
    dtype pull_1_ = packed_shfl_sync(MASK_ALL, accum_packed_1, leader_step_2);
    dtype pull_3_ = packed_shfl_sync(MASK_ALL, accum_packed_3, leader_step_2);
    if ( row % 4 == 2 || row % 4 == 3 ) {  accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0_); }
    if ( row % 4 == 2 || row % 4 == 3 ) {  accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2_); }
    if ( row % 4 == 2 || row % 4 == 3 ) {  accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1_); }
    if ( row % 4 == 2 || row % 4 == 3 ) {  accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3_); }
    __syncthreads();

    // step 3: each row pulls from the last row of the prior four rows
    int leader_step_3; 
    if ( row % 8 > 3 ) { 
        leader_step_3 = laneid() - 4*((row % 4) + 1);
    } else {  
        leader_step_3 = laneid();  
    }
    __syncthreads();
    dtype pull_0__ = packed_shfl_sync(MASK_ALL, accum_packed_0, leader_step_3);
    dtype pull_2__ = packed_shfl_sync(MASK_ALL, accum_packed_2, leader_step_3);
    dtype pull_1__ = packed_shfl_sync(MASK_ALL, accum_packed_1, leader_step_3);
    dtype pull_3__ = packed_shfl_sync(MASK_ALL, accum_packed_3, leader_step_3);
    if ( row % 8 > 3 ) {  accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0__); }
    if ( row % 8 > 3 ) {  accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2__); }
    if ( row % 8 > 3 ) {  accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1__); }
    if ( row % 8 > 3 ) {  accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3__); }
    __syncthreads();

    // step 4: each core matrix sends its last row to the ``next'' core matrix (0 to 1, 2 to 3)
    int leader_step_4; 
    if ( row % 8 < 7 ) { leader_step_4 = laneid() + 4*(7 - row);  }
    else { leader_step_4 = laneid(); }
    // 0 to 1
    row_vec<rt_bf<1,1>> last_row_0, last_row_2;
    rt_bf<1,1> broadcast_last_row_0, broadcast_last_row_2;
    dtype copy_accum_packed_0 = accum_packed_0;
    dtype copy_accum_packed_2 = accum_packed_2;

    copy_accum_packed_0 = packed_shfl_sync(MASK_ALL, copy_accum_packed_0, leader_step_4);
    last_row_0[0][0] = copy_accum_packed_0;
    broadcast_col(broadcast_last_row_0, last_row_0);
    dtype (*broadcast_last_row_0_) = reinterpret_cast<dtype*>(&broadcast_last_row_0);
    accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, *broadcast_last_row_0_);

    copy_accum_packed_2 = packed_shfl_sync(MASK_ALL, copy_accum_packed_2, leader_step_4);
    last_row_2[0][0] = copy_accum_packed_2;
    broadcast_col(broadcast_last_row_2, last_row_2);
    dtype (*broadcast_last_row_2_) = reinterpret_cast<dtype*>(&broadcast_last_row_2);
    accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, *broadcast_last_row_2_);
    __syncthreads();

    accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, *broadcast_last_row_1_);
    accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, *broadcast_last_row_1_);
    accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, *broadcast_last_row_3_);
    accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, *broadcast_last_row_3_);
    __syncthreads();

    dst.tiles[0][0].data[0] = accum_packed_0; 
    dst.tiles[0][0].data[2] = accum_packed_2;
    dst.tiles[0][0].data[1] = accum_packed_1; 
    dst.tiles[0][0].data[3] = accum_packed_3;
    __syncthreads();
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(
    int n, CUtensorMap* tma_k, CUtensorMap* debug_cumsum_k_a1
) {
    int laneid = kittens::laneid(); // who am i? when am i?
    int warpid = kittens::warpid(); // who am i? when am i?
    int tic = 0, toc = 1;

    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.x;
    const int batch_head_id = batch_id*gridDim.x + head_id;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf_1x1<wgmma_interleave_l> (&k_s) [2] = al.allocate<st_bf_1x1<wgmma_interleave_l>, 2>();       // 4096 bytes
    st_bf_1x1<wgmma_interleave_l> (&ks_smem_a1_tile) = al.allocate<st_bf_1x1<wgmma_interleave_l>>();  // 1024 bytes
    warpgroup::zero(ks_smem_a1_tile);

    rt_bf_1x1<> k_cumsum_a1_reg;
    zero(k_cumsum_a1_reg);

    int n_blocks = n / (k_s[0].rows);

    // initial load
    __shared__ tma::barrier bar;
    if (warpid == 0) tma::init_barrier(bar);
    __syncthreads();
    if (warpid == 0) {
        tma::set_bytes(bar, size_bytes<typeof(k_s[0])> );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(k_s[tic],   tma_k,   bar, tile_idx);
    }
    
    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        // arrive memory
        tma::arrive_and_wait(bar, tic);
        __syncthreads(); // everybody on the same page?
        if (warpid == 0 && block+1<n_blocks) {   // go get the next K from HBM
            tma::set_bytes(bar, size_bytes<typeof(k_s[0])>  );
            int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
            tma::load_async(k_s[toc],   tma_k,   bar, next_tile_idx);
        }

        rt_bf_1x1<> k_src;
        load(k_src, k_s[tic]);
        __syncthreads();
        cumulative_add(k_cumsum_a1_reg, k_src, block);  // cumsum in registers
        __syncthreads();
        __syncthreads();
        store(ks_smem_a1_tile, k_cumsum_a1_reg);
        __syncthreads();

        // __syncthreads(); 
        // cumulative_add(ks_smem_a1_tile, k_s[tic]);   // cumsum in smem
        // __syncthreads();
        // __syncthreads(); 
        if (warpid == 0) { 
            tma::store_async(debug_cumsum_k_a1, ks_smem_a1_tile, blockIdx.x*n_blocks + block); 
            tma::store_commit_group();   
        }
        tma::store_async_wait();
    }
}

#include "harness.impl"

