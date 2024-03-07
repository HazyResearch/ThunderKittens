/**
 * @file
 * @brief Warpgroup functions for transferring data directly between shared memory and registers and back.
 */



#pragma once

#include <type_traits>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace warpgroup {

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from shared memory to register tiles with optional type conversion.
 *
 * @tparam T2 The data type of the elements in the register tile.
 * @tparam U The data type of the elements in the shared memory tile.
 * @tparam height The height of the tile.
 * @tparam width The width of the tile.
 * @tparam reg_layout The layout of the register tile (row or column major).
 * @tparam shared_layout The layout of the shared memory tile (row major).
 * @param dst[out] The destination register tile.
 * @param src[in] The source shared memory tile.
 */
template<typename T2, typename U, int height, int width, rt_layout reg_layout, st_row_layout shared_layout>
__device__ inline static void load(rt<T2, height/4, width, reg_layout> &dst, const st<U, height, width, shared_layout> &src) {
    static_assert(height%4 == 0, "Warpgroup load / store requires tile height to be a multiple of 4.");
    constexpr int warp_height = height/4;
    int warpid = (threadIdx.x % 128) / 32;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U >::packed_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = (warpid*warp_height + i)*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+0}]));
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+0}]));
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+8}]));
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+8}]));
            }
            else {
                // handle the column-major layout
                int row = (warpid*warp_height + i)*dst.tile_size + 2*(laneid % 4);
                int col = j*dst.tile_size + (laneid / 4);
                dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src[{row+0, col+0}]);
                dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src[{row+1, col+0}]);
                dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src[{row+0, col+8}]);
                dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src[{row+1, col+8}]);
                dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src[{row+8, col+0}]);
                dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src[{row+9, col+0}]);
                dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src[{row+8, col+8}]);
                dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src[{row+9, col+8}]);
            }
        }
    }
}

/**
 * @brief Store data from register tiles to shared memory with optional type conversion.
 *
 * @tparam U The data type of the elements in the shared memory tile.
 * @tparam T2 The data type of the elements in the register tile.
 * @tparam height The height of the tile.
 * @tparam width The width of the tile.
 * @tparam reg_layout The layout of the register tile (row or column major).
 * @tparam shared_layout The layout of the shared memory tile (row major).
 * @param dst[out] The destination shared memory tile.
 * @param src[in] The source register tile.
 */
template<typename U, typename T2, int height, int width, rt_layout reg_layout, st_row_layout shared_layout>
__device__ inline static void store(st<U, height, width, shared_layout> &dst, const rt<T2, height/4, width, reg_layout> &src) {
    static_assert(height%4 == 0, "Warpgroup load / store requires tile height to be a multiple of 4.");
    constexpr int warp_height = height/4;
    int warpid = (threadIdx.x % 128) / 32;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < warp_height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = (warpid*warp_height + i)*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                *(U2*)(&dst[{row+0, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                *(U2*)(&dst[{row+8, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                *(U2*)(&dst[{row+0, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                *(U2*)(&dst[{row+8, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            }
            else {
                // handle the column-major layout
                int row = (warpid*warp_height + i)*src.tile_size + 2*(laneid % 4);
                int col = j*src.tile_size + (laneid / 4);
                dst[{row+0, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst[{row+1, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst[{row+0, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst[{row+1, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
                dst[{row+8, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
                dst[{row+9, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
                dst[{row+8, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
                dst[{row+9, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
            }
        }
    }
}

}
}
