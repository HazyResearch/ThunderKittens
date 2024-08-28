/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

// ----------- ROW LAYOUTS ----------

/**
 * @brief Loads data from global memory into a shared memory tile with a row layout.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 */
template<ducks::st::all ST>
__device__ static inline void load(ST &dst, const typename ST::dtype *src, const int row_stride) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height*dst.width * TILE_DIM*TILE_DIM / (WARP_THREADS*elem_per_memcpy);

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        *(float4*)(&dst[{row, col}]) = *(float4*)(&src[row*row_stride + col]);
    }
}
/**
 * @brief Loads data from global memory into a complex shared memory tile with a row layout.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] dst The destination complex shared memory tile.
 * @param[in] resrc The source global memory array for the real component.
 * @param[in] imsrc The source global memory array for the imaginary component.
 * @param re_row_stride[in] The stride between rows in the source real component array.
 * @param im_row_stride[in] The stride between rows in the source imaginary component array.
 */
template<ducks::st::complex CST>
__device__ static inline void load(CST &dst, const typename CST::dtype::dtype *resrc, const typename CST::dtype::dtype *imsrc, const int re_row_stride, const int im_row_stride) {
    load(dst.real, resrc, re_row_stride);
    load(dst.imag, imsrc, im_row_stride);
}
/**
 * @brief Stores data from a shared memory tile with a row layout into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<ducks::st::all ST>
__device__ static inline void store(typename ST::dtype *dst, const ST &src, const int row_stride) {

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height*src.width * TILE_DIM*TILE_DIM / (WARP_THREADS*elem_per_memcpy);

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        *(float4*)(&dst[row*row_stride + col]) = *(float4*)(&src[{row, col}]);
    }
}

/**
 * @brief Stores bf16 data from a complex shared memory tile with a row layout into global memory.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] redst The destination global memory array for the real component.
 * @param[out] imdst The destination global memory array for the imaginary component.
 * @param[in] src The source complex shared memory tile.
 * @param re_row_stride[in] The stride between rows in the destination real component array.
 * @param im_row_stride[in] The stride between rows in the destination imaginary component array.
 */
template<ducks::st::complex CST>
__device__ static inline void store(const typename CST::dtype::dtype *redst, const typename CST::dtype::dtype *imdst, CST &src, const int re_row_stride, const int im_row_stride) {
    store(redst, src.real, re_row_stride);
    store(imdst, src.imag, im_row_stride);
}
/**
 * @brief Asynchronously loads data from global memory into a shared memory tile with a row layout using CUDA barriers.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::st::all ST>
__device__ static inline void load_async(ST &dst, const typename ST::dtype *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height*dst.width * TILE_DIM*TILE_DIM / (WARP_THREADS*elem_per_memcpy);

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
 * @brief Asynchronously loads data from global memory into a complex shared memory tile with a row layout using CUDA barriers.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] resrc The source global memory array for the real component.
 * @param[in] imsrc The source global memory array for the imaginary component.
 * @param re_row_stride[in] The stride between rows in the real component source array.
 * @param im_row_stride[in] The stride between rows in the imaginary component source array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::st::complex CST>
__device__ static inline void load_async(CST &dst, const typename CST::dtype::dtype *resrc, const typename CST::dtype::dtype *imsrc, const int re_row_stride, const int im_row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    load_async(dst.real, resrc, re_row_stride, barrier);
    load_async(dst.imag, imsrc, im_row_stride, barrier);
}
/**
 * @brief Asynchronously stores data from a shared memory tile with a row layout into global memory using CUDA barriers.
 *
 * @tparam ST The type of the shared tile
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::st::all ST>
__device__ static inline void store_async(typename ST::dtype *dst, const ST &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height*src.width * TILE_DIM*TILE_DIM / (WARP_THREADS*elem_per_memcpy);

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

/**
 * @brief Asynchronously stores data from a complex shared memory tile with a row layout into global memory using CUDA barriers.
 *
 * @tparam ST The type of the shared tile
 * @param[out] redst The destination real component global memory array.
 * @param[out] imdst The destination imaginary component global memory array.
 * @param[in] src The source shared memory tile.
 * @param re_row_stride[in] The stride between rows in the real component destination array.
 * @param im_row_stride[in] The stride between rows in the imaginary component destination array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::st::complex CST>
__device__ static inline void store_async(typename CST::dtype::dtype *redst, typename CST::dtype::dtype *imdst, const CST &src, const int re_row_stride, const int im_row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    store_async(redst, src.real, re_row_stride, barrier);
    store_async(imdst, src.imag, im_row_stride, barrier);
}

/**
 * @brief Asynchronously loads bf16 data from global memory into a shared memory vec with a row layout using CUDA barriers.
 * @tparam SV The type of the shared vec.
 * @param[out] dst The destination shared memory vec.
 * @param[in] src The source global memory array.
 * @param row_stride[in] The stride between rows in the source array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::sv::all SV>
__device__ static inline void load_async(SV &dst, const bf16 *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = ::kittens::laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            cuda::memcpy_async(
                (void*)(&dst[i*elem_per_transfer]),
                (void*)(&src[i*elem_per_transfer]),
                cuda::aligned_size_t<16>(sizeof(float4)),
                barrier
            );
        }
    }
}

/**
 * @brief Asynchronously stores bf16 data from a shared memory vec with a row layout into global memory using CUDA barriers.
 *
 * @tparam SV The type of the shared vec
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vec.
 * @param row_stride[in] The stride between rows in the destination array.
 * @param barrier[in,out] The CUDA barrier used for synchronization.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<ducks::sv::all SV>
__device__ static inline void store_async(bf16 *dst, const SV &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = ::kittens::laneid(); i < total_calls; i+=WARP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            cuda::memcpy_async(
                (void*)(&dst[i*elem_per_transfer]),
                (void*)(&src[i*elem_per_transfer]),
                cuda::aligned_size_t<16>(sizeof(float4)),
                barrier
            );
        }
    }
}


}