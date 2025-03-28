/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;

    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row_0to3 = i*dst.tile_size_row + (warphalflaneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + warphalf*8 + 2*(laneid % 4);
            T2 transfers[2];
            transfers[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+0)*row_stride + col]));
            transfers[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+4)*row_stride + col]));
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            dst.tiles[i][j].data[0] = transfers[0];
            dst.tiles[i][j].data[2] = transfers[1];
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + warphalf*8 + 2*(laneid % 4);
            T2 transfers[2];
            transfers[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+ 8)*row_stride + col]));
            transfers[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+12)*row_stride + col]));
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            dst.tiles[i][j].data[1] = transfers[0];
            dst.tiles[i][j].data[3] = transfers[1];
        }
    }
}
/**
 * @brief Load data from a source array into a column-major layout tile.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    #ifdef KITTENS_HOPPER
    // static assert that we're not using fp8e4m3 or fp8e5m2
    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<T, fp8e5m2>, "Unsupported type for load/store");
    #endif
    
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size_row + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (laneid / 4);
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (laneid / 4);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (laneid / 4);
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (laneid / 4);
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+8)]);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;

    #ifdef KITTENS_HOPPER
    // static assert that we're not using fp8e4m3 or fp8e5m2
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row_0to3 = i*src.tile_size_row + (warphalflaneid / 4);
        int row = i*src.tile_size_row + (laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            *(U2*)(&dst_ptr[(row_0to3+0)*row_stride + col]) = transfers[0];
            *(U2*)(&dst_ptr[(row_0to3+4)*row_stride + col]) = transfers[1];
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            *(U2*)(&dst_ptr[(row_0to3+ 8)*row_stride + col]) = transfers[0];
            *(U2*)(&dst_ptr[(row_0to3+12)*row_stride + col]) = transfers[1];
        }
    }
}
/**
 * @brief Store data from a register tile to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    #ifdef KITTENS_HOPPER
    // static assert that we're not using fp8e4m3 or fp8e5m2
    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<T, fp8e5m2>, "Unsupported type for load/store");
    #endif

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            dst_ptr[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            dst_ptr[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            dst_ptr[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            dst_ptr[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}