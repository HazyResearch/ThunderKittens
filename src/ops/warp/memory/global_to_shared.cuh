#pragma once

#include <cuda/pipeline>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// ----------- ROW LAYOUTS ----------

template<ducks::st::row_layout ST>
__device__ static inline void load(ST &dst, const bf16 *src, const int row_stride) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        *(float4*)(&dst[{row, col}]) = *(float4*)(&src[row*row_stride + col]);
    }
}
template<ducks::st::row_layout ST>
__device__ static inline void store(bf16 *dst, const ST &src, const int row_stride) {

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        *(float4*)(&dst[row*row_stride + col]) = *(float4*)(&src[{row, col}]);
    }
}

template<ducks::st::row_layout ST>
__device__ static inline void load_async(ST &dst, const bf16 *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        cuda::memcpy_async(
            (void*)(&dst[{row, col}]),
            (void*)(&src[row*row_stride + col]),
            cuda::aligned_size_t<16>(sizeof(float4)),
            barrier
        );
    }
}
template<ducks::st::row_layout ST>
__device__ static inline void store_async(bf16 *dst, const ST &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        cuda::memcpy_async(
            (void*)(&dst[row*row_stride + col]),
            (void*)(&src[{row, col}]),
            cuda::aligned_size_t<16>(sizeof(float4)),
            barrier
        );
    }
}


// ----------- COL LAYOUTS ----------

template<ducks::st::col_layout ST>
__device__ static inline void load(ST &dst, const bf16 *src, const int row_stride) {
    
    int laneid = threadIdx.x % 32;

    // in column mode we unfortunately can only transfer one element at at time.
    int elem_per_memcpy = 1;
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width * 8;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        dst[{row, col}] = src[row*row_stride + col];
    }
}
template<ducks::st::col_layout ST>
__device__ static inline void store(bf16 *dst, const ST &src, const int row_stride) {

    int laneid = threadIdx.x % 32;

    // in column mode we unfortunately can only transfer one element at at time.
    int elem_per_memcpy = 1;
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width * 8;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        dst[row*row_stride + col] = src[{row, col}];
    }
}

// ----------  VECTORS ----------
template<ducks::sv::all SV>
__device__ static inline void load(SV &dst, const typename SV::dtype *src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer];
    }
}
template<ducks::sv::all SV>
__device__ static inline void store(typename SV::dtype *dst, const SV &src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < src.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer]; // lmao it's identical
    }
}

}