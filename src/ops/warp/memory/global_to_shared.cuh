#pragma once

#include <cuda/pipeline>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// ----------- ROW LAYOUTS ----------

/**
 * @brief Loads bf16 data from global memory into a shared memory tile with a row layout.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory row layout.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride The stride between rows in the source array.
 */
template<int height, int width, st_row_layout layout>
__device__ static inline void load(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride) {
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

/**
 * @brief Stores bf16 data from a shared memory tile with a row layout into global memory.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory row layout.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride The stride between rows in the destination array.
 */
template<int height, int width, st_row_layout layout>
__device__ static inline void store(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride) {

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

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @tparam U The data type of the source global memory array.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 */
template<st_vec_type ST, typename U>
__device__ inline static void load(ST &dst, const U *src) {

    int laneid = threadIdx.x % 32;
    int total_calls = (dst.length+31)/32; // allowing it to round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int idx = i * 32 + laneid;
        if (idx < dst.length) { dst[idx] = src[idx]; }
    }
}

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @tparam U The data type of the destination global memory array.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 */
template<st_vec_type ST, typename U>
__device__ inline static void store(U *dst, const ST &src) {

    int laneid = threadIdx.x % 32;
    int total_calls = (src.length+31)/32; // allowing it to round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int idx = i * 32 + laneid;
        if (idx < src.length) { dst[idx] = src[idx]; }
    }
}

/**
 * @brief Asynchronously loads bf16 data from global memory into a shared memory tile with a row layout using CUDA barriers.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory row layout.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride The stride between rows in the source array.
 * @param barrier The CUDA barrier used for synchronization.
 */
template<int height, int width, st_row_layout layout>
__device__ static inline void load_async(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
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

/**
 * @brief Asynchronously stores bf16 data from a shared memory tile with a row layout into global memory using CUDA barriers.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory row layout.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride The stride between rows in the destination array.
 * @param barrier The CUDA barrier used for synchronization.
 */
template<int height, int width, st_row_layout layout>
__device__ static inline void store_async(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
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

/**
 * @brief Loads bf16 data from global memory into a shared memory tile with a column layout.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory column layout.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride The stride between rows in the source array.
 */
template<int height, int width, st_col_layout layout>
__device__ static inline void load(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride) {

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

/**
 * @brief Stores bf16 data from a shared memory tile with a column layout into global memory.
 *
 * @tparam height The number of rows in the tile.
 * @tparam width The number of columns in the tile.
 * @tparam layout The shared memory column layout.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride The stride between rows in the destination array.
 */
template<int height, int width, st_col_layout layout>
__device__ static inline void store(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride) {

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

}
