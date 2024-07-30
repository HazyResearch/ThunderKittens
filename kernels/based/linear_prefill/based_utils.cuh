#include "kittens.cuh"

using namespace kittens;

#define tile_k_reg rt_bf<4,1>
#define tile_k_reg_fl rt_fl<4,1>
#define tile_k_reg_loads rt_bf<1,1>
#define tile_k_reg_loads_fl rt_fl<1,1>

// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
__device__ void accumulate_a0(sv_bf<ST::width> &a0_total, const ST &v) {
    int col = threadIdx.x*2;
    constexpr int PARALLEL_LOADS = 16;
    float2 v_data[PARALLEL_LOADS];
    if(col < ST::cols) {
        float2 acc = __bfloat1622float2(*(bf16_2*)&a0_total[col]);
        #pragma unroll
        for(int k = 0; k < ST::rows/PARALLEL_LOADS; k++) {
            #pragma unroll
            for(int i = 0; i < PARALLEL_LOADS; i++) { // load it all in
                int row = k*PARALLEL_LOADS+i;
                v_data[i] = __bfloat1622float2(*(bf16_2*)&v[int2{row, col}]);
            }
            #pragma unroll
            for(int i = 0; i < PARALLEL_LOADS; i++) { // accumulate, through registers, and write
                acc.x += v_data[i].x;
                acc.y += v_data[i].y;
            }
        }
        *(bf16_2*)&a0_total[col] = __float22bfloat162_rn(acc);
    }
}

// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col+i].unsqueeze(0) for i in range(4)], dim=-1)
__device__ static void mul_slice_row(rt_bf_1x4<> &dst, const rt_bf_1x1<> &src, const int starting_col) {
    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<rt_bf_1x1<>&>(dst.tiles[0][i]), src);
        const int target_col = starting_col + i;
        #pragma unroll
        for(int row_offset = 0; row_offset < 2; row_offset++) {
            const int src_thread = (lane / 4)*4 + (target_col%8)/2;
            const int col_offset = target_col >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[row_offset] *= bf16_2{val, val};
            dst.tiles[0][i].data[row_offset+2] *= bf16_2{val, val};
        }
    }
}


// in pytorch, this computes, for a 16x64 tensor dst and 16x64 tensor src:
// dst = src * src[:,starting_col].unsqueeze(-1)
__device__ static void mul_slice_col(rt_bf_1x4<> &dst, const rt_bf_1x4<> &src, const int target_row) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two cols
    copy(dst, src);
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int col_offset = 0; col_offset < 2; col_offset++) {
            const int src_thread = (target_row%8)*4 + (lane%4);
            const int row_offset = target_row >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            bf16_2 val = __shfl_sync(kittens::MASK_ALL, src_val, src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[col_offset*2+0] *= val;
            dst.tiles[0][i].data[col_offset*2+1] *= val;
        }
    }
}

template<ducks::sv::all SV, ducks::st::all ST>
__device__ inline void cumulative_add(SV &dst, const ST &src) {
    // this is called along a warpgroup
    static_assert(ST::cols <= 128);
    static_assert(ST::cols == SV::length);
    int lane = threadIdx.x % 128;
    if(lane < ST::cols) {
        float f = __bfloat162float(dst[lane]);
        // acc equal to the last row of dst
        for (auto i = 0; i < ST::rows; i++) {
            f += __bfloat162float(src[{i, lane}]);
        }
        dst[lane] = __float2bfloat16(f);
    }
}

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

template<ducks::sv::all SV, ducks::st::all ST>
__device__ inline void cumulative_add_a0(SV &dst, const ST &src) {
    // this is called along a warpgroup
    static_assert(ST::cols <= 128);
    static_assert(ST::cols == SV::length);
    int lane = threadIdx.x % 128;
    if(lane < ST::cols) {
        float f = __bfloat162float(dst[lane]);
        // acc equal to the last row of dst
        for (auto i = 0; i < ST::rows; i++) {
            f += i;
        }
        dst[lane] = __float2bfloat16(f);
    }
}

template<ducks::rt::all RT>
__device__ inline void cumulative_add(RT &dst, const RT &src, const int block_idx) {
    using dtype = typename RT::dtype;
    using packed_type = typename base_types::packing<dtype>::unpacked_type;
    // we know that src and dst will have the same dtype and shape by definition
    static_assert(dst.width == 1); // hard coded to this case

    // get the leaders for different shuffles.
    const int row = laneid() / 4;
    
    int leader_step_0; 
    if ( row % 8 < 7 ) { leader_step_0 = laneid() + 4*(7 - row);  }
    else { leader_step_0 = laneid(); }

    const int leader_step_1 = (row % 2 == 1) ? laneid() - 4: laneid();

    int leader_step_2;
    if (row % 4 == 2 || row % 4 == 3 ) { leader_step_2 = (row % 2 == 0) ? laneid() - 4 : laneid() - 8; } 
    else {  leader_step_2 = laneid(); }  

    int leader_step_3; 
    if ( row % 8 > 3 ) { leader_step_3 = laneid() - 4*((row % 4) + 1); } 
    else { leader_step_3 = laneid(); }

    int leader_step_4; 
    if ( row % 8 < 7 ) { leader_step_4 = laneid() + 4*(7 - row);  }
    else { leader_step_4 = laneid(); }

    // Extract the last row of dst from the prior block.
    row_vec<tile_k_reg_loads_fl> last_row_1, last_row_3;
    tile_k_reg_loads_fl broadcast_last_row_1, broadcast_last_row_3;
    dtype copy_accum_packed_1 = dst.tiles[dst.height-1][0].data[1];
    dtype copy_accum_packed_3 = dst.tiles[dst.height-1][0].data[3];

    copy_accum_packed_1 = packed_shfl_sync(MASK_ALL, copy_accum_packed_1, leader_step_0);
    last_row_1[0][0] = copy_accum_packed_1;
    broadcast_col(broadcast_last_row_1, last_row_1);
    dtype (*broadcast_last_row_1_) = reinterpret_cast<dtype*>(&broadcast_last_row_1);

    copy_accum_packed_3 = packed_shfl_sync(MASK_ALL, copy_accum_packed_3, leader_step_0);
    last_row_3[0][0] = copy_accum_packed_3;
    broadcast_col(broadcast_last_row_3, last_row_3);
    dtype (*broadcast_last_row_3_) = reinterpret_cast<dtype*>(&broadcast_last_row_3);
    __syncwarp();

    // initial values
    dtype accum_packed_0 = src.tiles[0][0].data[0];
    dtype accum_packed_1 = src.tiles[0][0].data[1];
    dtype accum_packed_2 = src.tiles[0][0].data[2];
    dtype accum_packed_3 = src.tiles[0][0].data[3];
    dtype final_0, final_1, final_2, final_3;

    // start the cumsum
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        
        dtype (*broadcast_last_row_inner_1_), (*broadcast_last_row_inner_3_);
        if (i > 0) {
            // last row of the previous core matrix from src tiles.
            row_vec<tile_k_reg_loads_fl> last_row_inner_1, last_row_inner_3;
            tile_k_reg_loads_fl broadcast_last_row_inner_1, broadcast_last_row_inner_3;
            dtype copy_accum_packed_inner_1 = accum_packed_1;
            dtype copy_accum_packed_inner_3 = accum_packed_3;

            copy_accum_packed_inner_1 = packed_shfl_sync(MASK_ALL, copy_accum_packed_inner_1, leader_step_0);
            last_row_inner_1[0][0] = copy_accum_packed_inner_1;
            broadcast_col(broadcast_last_row_inner_1, last_row_inner_1);
            broadcast_last_row_inner_1_ = reinterpret_cast<dtype*>(&broadcast_last_row_inner_1);

            copy_accum_packed_inner_3 = packed_shfl_sync(MASK_ALL, copy_accum_packed_inner_3, leader_step_0);
            last_row_inner_3[0][0] = copy_accum_packed_inner_3;
            broadcast_col(broadcast_last_row_inner_3, last_row_inner_3);
            broadcast_last_row_inner_3_ = reinterpret_cast<dtype*>(&broadcast_last_row_inner_3);
            __syncwarp();

            // add it to the starting point.
            accum_packed_0 = src.tiles[i][0].data[0]; 
            accum_packed_1 = src.tiles[i][0].data[1]; 
            accum_packed_2 = src.tiles[i][0].data[2]; 
            accum_packed_3 = src.tiles[i][0].data[3]; 
        }
        __syncwarp();

        // step 1. even row i ships its values to odd row i+1
        dtype pull_0 = packed_shfl_sync(MASK_ALL, accum_packed_0, leader_step_1);
        dtype pull_1 = packed_shfl_sync(MASK_ALL, accum_packed_1, leader_step_1);
        dtype pull_2 = packed_shfl_sync(MASK_ALL, accum_packed_2, leader_step_1);
        dtype pull_3 = packed_shfl_sync(MASK_ALL, accum_packed_3, leader_step_1);
        if ( row % 2 == 1) {  
            accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0); 
            accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2); 
            accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1); 
            accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3); 
        }
        __syncwarp();

        // step 2. each row pulls from the last row of the prior two rows cumsum
        dtype pull_0_ = packed_shfl_sync(MASK_ALL, accum_packed_0, leader_step_2);
        dtype pull_2_ = packed_shfl_sync(MASK_ALL, accum_packed_2, leader_step_2);
        dtype pull_1_ = packed_shfl_sync(MASK_ALL, accum_packed_1, leader_step_2);
        dtype pull_3_ = packed_shfl_sync(MASK_ALL, accum_packed_3, leader_step_2);
        if ( row % 4 == 2 || row % 4 == 3 ) {  
            accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0_); 
            accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2_); 
            accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1_); 
            accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3_); 
        }
        __syncwarp();

        // step 3: each row pulls from the last row of the prior four rows
        dtype pull_0__ = packed_shfl_sync(MASK_ALL, accum_packed_0, leader_step_3);
        dtype pull_2__ = packed_shfl_sync(MASK_ALL, accum_packed_2, leader_step_3);
        dtype pull_1__ = packed_shfl_sync(MASK_ALL, accum_packed_1, leader_step_3);
        dtype pull_3__ = packed_shfl_sync(MASK_ALL, accum_packed_3, leader_step_3);
        if ( row % 8 > 3 ) {  
            accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, pull_0__); 
            accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, pull_2__); 
            accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, pull_1__); 
            accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, pull_3__); 
        }
        __syncwarp();

        // step 4: each core matrix sends its last row to the ``next'' core matrix (0 to 1, 2 to 3)
        row_vec<tile_k_reg_loads_fl> last_row_0, last_row_2;
        tile_k_reg_loads_fl broadcast_last_row_0, broadcast_last_row_2;
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
        __syncwarp();

        // Finally, save everything out.
        final_0 = base_ops::sum::op<dtype>(accum_packed_0, *broadcast_last_row_1_);
        final_1 = base_ops::sum::op<dtype>(accum_packed_1, *broadcast_last_row_1_);
        final_2 = base_ops::sum::op<dtype>(accum_packed_2, *broadcast_last_row_3_);
        final_3 = base_ops::sum::op<dtype>(accum_packed_3, *broadcast_last_row_3_);
        if (i > 0) {
            final_0 = base_ops::sum::op<dtype>(final_0, *broadcast_last_row_inner_1_);
            final_1 = base_ops::sum::op<dtype>(final_1, *broadcast_last_row_inner_1_);
            final_2 = base_ops::sum::op<dtype>(final_2, *broadcast_last_row_inner_3_);
            final_3 = base_ops::sum::op<dtype>(final_3, *broadcast_last_row_inner_3_);
            accum_packed_0 = base_ops::sum::op<dtype>(accum_packed_0, *broadcast_last_row_inner_1_);
            accum_packed_1 = base_ops::sum::op<dtype>(accum_packed_1, *broadcast_last_row_inner_1_);
            accum_packed_2 = base_ops::sum::op<dtype>(accum_packed_2, *broadcast_last_row_inner_3_);
            accum_packed_3 = base_ops::sum::op<dtype>(accum_packed_3, *broadcast_last_row_inner_3_);
        }
        __syncwarp();
        dst.tiles[i][0].data[0] = final_0;
        dst.tiles[i][0].data[1] = final_1;
        dst.tiles[i][0].data[2] = final_2;
        dst.tiles[i][0].data[3] = final_3;
        __syncwarp();
    }
}