/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include <cuda/pipeline>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::st COORD, int N_THREADS=WARP_THREADS>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int total_calls = (dst.height*dst.width * TILE_DIM*TILE_DIM + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<typename COORD::BASE, axis>(idx);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            if (row + idx.dim<axis>() < src.shape<axis>()) {
                float4 tmp;
                move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
            else {
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void load(ST &dst, const GL &src, const coord<ST> &idx) {
    load<2, false, ST, GL, coord<ST>, WARP_THREADS>(dst, src, idx);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::st COORD, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    const int row_stride = dst.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int total_calls = (src.height*src.width * TILE_DIM*TILE_DIM + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst.template get<ST, axis>(idx);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
        }
        else {
            if (row + idx.dim<axis>() < dst.shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store(const GL &dst, const ST &src, const coord<ST> &idx) {
    store<2, false, ST, GL, coord<ST>, WARP_THREADS>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::st COORD, int N_THREADS=WARP_THREADS>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int total_calls = (dst.height*dst.width * TILE_DIM*TILE_DIM + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<ST, axis>(idx);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * N_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if constexpr (assume_aligned) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
        else {
            if (row + idx.dim<axis>() < src.shape<axis>()) {
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                    : "memory"
                );
            }
            else {
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

}