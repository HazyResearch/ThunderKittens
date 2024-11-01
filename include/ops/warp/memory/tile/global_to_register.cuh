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
template<ducks::rt::row_layout RT, ducks::gt::l::all GTL, int axis=2>
__device__ inline static void load(RT &dst, const GTL &src, const index &idx) {
    ducks::g::check_raw<GTL, RT>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    using T2 = RT::dtype;
    using U = typename GTL::dtype;
    U *src_ptr = (U*)&src[idx];
    const int row_stride = src.stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row_0to3 = i*dst.tile_size + (warphalflaneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + warphalf*8 + 2*(laneid % 4);
            T2 transfers[2];
            transfers[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+0)*row_stride + col]));
            transfers[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row_0to3+4)*row_stride + col]));
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            dst.tiles[i][j].data[0] = transfers[0];
            dst.tiles[i][j].data[2] = transfers[1];
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + warphalf*8 + 2*(laneid % 4);
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
template<ducks::rt::col_layout RT, ducks::gt::l::all GTL, int axis=2>
__device__ inline static void load(RT &dst, const GTL &src, const index &idx) {
    ducks::g::check_raw<GTL, RT>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GTL::dtype;
    U *src_ptr = (U*)&src[idx];
    const int row_stride = src.stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+8)]);
        }
    }
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
template<ducks::rt::row_layout RT, ducks::gt::l::all GTL, int axis=2>
__device__ inline static void store(GTL &dst, const RT &src, const index &idx) {
    ducks::g::check_raw<GTL, RT>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    using T2 = RT::dtype;
    using U = typename GTL::dtype;
    U *dst_ptr = (U*)&dst[idx];
    const int row_stride = dst.stride<axis>();
    using U2 = base_types::packing<U>::packed_type;
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row_0to3 = i*src.tile_size + (warphalflaneid / 4);
        int row = i*src.tile_size + (laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            *(U2*)(&dst_ptr[(row_0to3+0)*row_stride + col]) = transfers[0];
            *(U2*)(&dst_ptr[(row_0to3+4)*row_stride + col]) = transfers[1];
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + warphalf*8 + 2*(laneid % 4);
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
template<ducks::rt::col_layout RT, ducks::gt::l::all GTL, int axis=2>
__device__ inline static void store(GTL &dst, const RT &src, const index &idx) {
    ducks::g::check_raw<GTL, RT>{}; // GTL must include a raw pointer to use non-TMA loads and stores
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GTL::dtype;
    U *dst_ptr = (U*)&dst[idx];
    const int row_stride = dst.stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst_ptr[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst_ptr[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst_ptr[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst_ptr[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}

}