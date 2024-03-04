#pragma once

#include <type_traits>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

// ----------  ROW LAYOUTS ----------

template<typename T2, typename U, int height, int width, rt_layout reg_layout, st_row_layout shared_layout>
__device__ inline static void load(rt<T2, height, width, reg_layout> &dst, const st<U, height, width, shared_layout> &src) {
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U >::packed_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+0}]));
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+0}]));
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+8}]));
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+8}]));
            }
            else {
                // handle the column-major layout
                int row = i*dst.tile_size + 2*(laneid % 4);
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


template<typename U, typename T2, int height, int width, rt_layout reg_layout, st_row_layout shared_layout>
__device__ inline static void store(st<U, height, width, shared_layout> &dst, const rt<T2, height, width, reg_layout> &src) {
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                *(U2*)(&dst[{row+0, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                *(U2*)(&dst[{row+8, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                *(U2*)(&dst[{row+0, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                *(U2*)(&dst[{row+8, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            }
            else {
                // handle the column-major layout
                int row = i*src.tile_size + 2*(laneid % 4);
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

// ----------  COL LAYOUTS ---------- (slow for the time being)

template<typename T2, typename U, int height, int width, rt_layout reg_layout, st_col_layout shared_layout>
__device__ inline static void load(rt<T2, height, width, reg_layout> &dst, const st<U, height, width, shared_layout> &src) {
    using T  = base_types::packing<T2>::unpacked_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src[{row+0, col+0}]);
                dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src[{row+0, col+1}]);
                dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src[{row+8, col+0}]);
                dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src[{row+8, col+1}]);
                dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src[{row+0, col+8}]);
                dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src[{row+0, col+9}]);
                dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src[{row+8, col+8}]);
                dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src[{row+8, col+9}]);
            }
            else {
                // handle the column-major layout
                int row = i*dst.tile_size + 2*(laneid % 4);
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

template<typename U, typename T2, int height, int width, rt_layout reg_layout, st_col_layout shared_layout>
__device__ inline static void store(st<U, height, width, shared_layout> &dst, const rt<T2, height, width, reg_layout> &src) {
    using T  = base_types::packing<T2>::unpacked_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (std::is_same_v<reg_layout, rt_row_layout>) {
                // handle the row-major layout
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                dst[{row+0, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst[{row+0, col+1}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst[{row+8, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst[{row+8, col+1}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
                dst[{row+0, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
                dst[{row+0, col+9}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
                dst[{row+8, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
                dst[{row+8, col+9}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
            }
            else {
                // handle the column-major layout
                int row = i*src.tile_size + 2*(laneid % 4);
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