/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace warpgroup {

// ----------- ROW LAYOUTS ----------

/**
 * @brief Collaboratively loads bf16 data from global memory into a shared memory tile with a row layout.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 */
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void load(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride) {
    // each thread needs to do 1 call per width*height/4
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 128;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = (dst.height * dst.width + 3) / 4; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 128 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows) // the first condition lets compiler short-circuit on unrolled iters
            *(float4*)(&dst[{row, col}]) = *(float4*)(&src[row*row_stride + col]);
    }
}
/**
 * @brief Collaboratively stores bf16 data from a shared memory tile with a row layout into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void store(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride) {

    int laneid = threadIdx.x % 128;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = (src.height * src.width + 3) / 4; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 128 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        if (i<total_calls-1 || row<src.rows)  // the first condition lets compiler short-circuit on unrolled iters
            *(float4*)(&dst[row*row_stride + col]) = *(float4*)(&src[{row, col}]);
    }
}


/**
 * @brief Collaborateively asynchronously loads bf16 data from global memory into a shared memory tile with a row layout using CUDA barriers.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void load_async(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height/4
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 128;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = (dst.height * dst.width + 3) / 4; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 128 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows)  // the first condition lets compiler short-circuit on unrolled iters
            cuda::memcpy_async(
                (void*)(&dst[{row, col}]),
                (void*)(&src[row*row_stride + col]),
                cuda::aligned_size_t<16>(sizeof(float4)),
                barrier
            );
    }
}
/**
 * @brief Collaboratively asynchronously stores bf16 data from a shared memory tile with a row layout into global memory using CUDA barriers.
 *
 * @tparam ST The type of the shared tile
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void store_async(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height/4
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 128;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = (src.height * src.width + 3) / 4; // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 128 + laneid;
        
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

// ----------  VECTORS ----------

/**
 * @brief Collaboratively loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 */
template<ducks::sv::all SV>
__device__ static inline void load(SV &dst, const typename SV::dtype *src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = laneid(); i < total_calls; i+=WARPGROUP_THREADS) {
        if(i * elem_per_transfer < dst.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer];
    }
}
/**
 * @brief Collaboratively stores data into a shared tile from global memory.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::sv::all SV>
__device__ static inline void store(typename SV::dtype *dst, const SV &src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = laneid(); i < total_calls; i+=WARPGROUP_THREADS) {
        if(i * elem_per_transfer < src.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer]; // lmao it's identical
    }
}

}
}