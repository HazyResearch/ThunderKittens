#pragma once

#include "kittens.cuh"

// KERNEL PARAMS

constexpr int NUM_WORKERS = 12; // hardcoded, don't change. 4 producer, 8 consumer
constexpr int NUM_THREADS = NUM_WORKERS*kittens::WARP_THREADS;

// TYPE UTILS

// these types are hardcoded into the kernel, but are nonetheless defined here for clarity.
using q_tile = kittens::st_bf_4x1;
using k_tile = kittens::st_bf_4x1;
using v_tile = kittens::st_bf_4x4;
using o_tile = kittens::st_bf_4x4;

using a0_vec        = kittens::sv_fl_4; // this one we may as well do in fp32
using a1_trans_tile = kittens::st_bf_4x1;
// using a2_tile_fl    = kittens::st_fl_4x4;
using a2_tile       = kittens::st_bf_4x4;

using kv_a0_tile = kittens::sv_fl_4;
using kv_a1_tile = kittens::st_fl_4x1;
using kv_a2_tile = kittens::st_fl_4x4;

// A0 UTILS

// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
__device__ void accumulate_a0(kittens::sv_fl<ST::width> &a0_total, const ST &v) {
    int col = threadIdx.x - 128; // this is executed by the second consumer warpgroup
    if(col < ST::cols) {
        float acc = a0_total[col];
        #pragma unroll
        for(int row = 0; row < ST::rows; row++) {
            kittens::bf16 val;
            kittens::move<kittens::bf16>::lds(val, &v[int2{row, col}]);
            acc += __bfloat162float(val);
        }
        kittens::move<float>::sts(&a0_total[col], acc);
    }
}

// we also need a function that adds a cumulative onto a register vector split across a warpgroup
__device__ inline void norm_add_cumsum_a0(kittens::rv<float, 1, 1> &a0_total) {
    int warp = kittens::warpid();
    int base_row = warp*16 + kittens::laneid()/4;
    a0_total[0][0].x += float(base_row);
    a0_total[0][0].y += float(base_row + 8);
}


// A1 UTILS


// A2 UTILS



// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col+i].unsqueeze(0) for i in range(4)], dim=-1)
__device__ static void mul_slice_row(kittens::rt_bf_1x4<> &dst, const kittens::rt_bf_1x1<> &src, const int starting_col) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<kittens::rt_bf_1x1<>&>(dst.tiles[0][i]), src);
        __syncwarp();
        const int target_col = starting_col + i;
        #pragma unroll
        for(int row_offset = 0; row_offset < 2; row_offset++) {
            const int src_thread = (lane / 4)*4 + (target_col%8)/2;
            const int col_offset = target_col >= 8;
            kittens::bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            __syncwarp();
            kittens::bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread); // correct value obtained and passed around
            __syncwarp();
            
            dst.tiles[0][i].data[row_offset] *= kittens::bf16_2{val, val};
            __syncwarp();
            dst.tiles[0][i].data[row_offset+2] *= kittens::bf16_2{val, val};
            __syncwarp();
        }
    }
}

// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col].unsqueeze(-1) for _ in range(4)], dim=-1)
__device__ static void mul_slice_col(kittens::rt_bf_1x4<> &dst, const kittens::rt_bf_1x4<> &src, const int target_row) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two cols
    copy(dst, src);
    __syncwarp();
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int col_offset = 0; col_offset < 2; col_offset++) {
            const int src_thread = (target_row%8)*4 + (lane%4);
            const int row_offset = target_row >= 8;
            // if(lane == 0) printf("target_row: %d, src_thread: %d, row_offset: %d, col_offset: %d\n", target_row, src_thread, row_offset, col_offset);
            kittens::bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            __syncwarp();
            kittens::bf16_2 val = __shfl_sync(kittens::MASK_ALL, src_val, src_thread); // correct value obtained and passed around
            __syncwarp();

            dst.tiles[0][i].data[col_offset*2+0] *= val;
            __syncwarp();
            dst.tiles[0][i].data[col_offset*2+1] *= val;
            __syncwarp();
        }
    }
}

// FOR DEBUG

#define RED  "\033[91m" 
#define GREEN  "\033[92m" 
#define YELLOW  "\033[93m" 
#define BLUE  "\033[94m" 
#define MAGENTA  "\033[95m" 
#define CYAN  "\033[96m" 
#define WHITE  "\033[97m" 
#define RESET  "\033[0m" 

template<typename... Args> __device__ void gprintf(Args... args) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x % 32 == 0) {
        printf(args...);
    }
}