#pragma once

#include <cuda/pipeline>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"


// ----------- ROW LAYOUTS ----------

template<ducks::st::row_layout ST>
__device__ static inline void load(ST &dst, const typename ST::dtype *src, const int row_stride) {
    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = (dst.height * dst.width + (N_WARPS-1)) / N_WARPS; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows) // the first condition lets compiler short-circuit on unrolled iters
            *(float4*)(&dst[{row, col}]) = *(float4*)(&src[row*row_stride + col]);
    }
}
template<ducks::st::row_layout ST>
__device__ static inline void store(typename ST::dtype *dst, const ST &src, const int row_stride) {

    int laneid = threadIdx.x;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = (src.height * src.width + (N_WARPS-1)) / N_WARPS; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        if (i<total_calls-1 || row<src.rows)  // the first condition lets compiler short-circuit on unrolled iters
            *(float4*)(&dst[row*row_stride + col]) = *(float4*)(&src[{row, col}]);
    }
}

template<ducks::st::row_layout ST>
__device__ static inline void load_async(ST &dst, const typename ST::dtype *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = (dst.height * dst.width + (N_WARPS-1)) / N_WARPS; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows) // the first condition lets compiler short-circuit on unrolled iters
            cuda::memcpy_async(
                (void*)(&dst[{row, col}]),
                (void*)(&src[row*row_stride + col]),
                cuda::aligned_size_t<16>(sizeof(float4)),
                barrier
            );
    }
}
template<ducks::st::row_layout ST>
__device__ static inline void store_async(typename ST::dtype *dst, const ST &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height/4
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = (src.height * src.width + (N_WARPS-1)) / N_WARPS; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        if (i<total_calls-1 || row<src.rows)  // the first condition lets compiler short-circuit on unrolled iters
            cuda::memcpy_async(
                (void*)(&dst[row*row_stride + col]),
                (void*)(&src[{row, col}]),
                cuda::aligned_size_t<16>(sizeof(float4)),
                barrier
            );
    }
}