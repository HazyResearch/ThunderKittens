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
template<ducks::st::all ST, ducks::gt::l::all GTL>
__device__ static inline void load(ST &dst, const GTL &src, const index &idx) {
    ducks::g::check_raw<GTL, ST>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    typename GTL::dtype *src_ptr = (typename GTL::dtype*)&src[idx];
    const int row_stride = src.row_stride();
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

        float4 tmp;
        move<float4>::ldg(tmp, &src_ptr[row*row_stride + col]);
        move<float4>::sts(&dst[{row, col}], tmp);
    }
}
/**
 * @brief Stores data from a shared memory tile with a row layout into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<ducks::st::all ST, ducks::gt::l::all GTL>
__device__ static inline void store(const GTL &dst, const ST &src, const index &idx) {
    ducks::g::check_raw<GTL, ST>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    typename GTL::dtype *dst_ptr = (typename GTL::dtype*)&dst[idx];
    const int row_stride = dst.row_stride();

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

        float4 tmp;
        move<float4>::lds(tmp, &src[{row, col}]);
        move<float4>::stg(&dst_ptr[row*row_stride + col], tmp);
    }
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
template<ducks::st::all ST, ducks::gt::l::all GTL>
__device__ static inline void load_async(ST &dst, GTL &src, const index &idx) {
    ducks::g::check_raw<GTL, ST>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    typename GTL::dtype *src_ptr = (typename GTL::dtype*)&src[idx];
    const int row_stride = src.row_stride();
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

        asm volatile(
            "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
            :
            : "l"((uint64_t)&dst[{row, col}]), "l"((uint64_t)&src_ptr[row*row_stride + col])
            : "memory"
        );
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N=0> __device__ static inline void load_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    __syncwarp();
}

}