#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

template<ducks::rt::row_layout RT, typename U>
__device__ inline static void load(RT &dst, const U *src, const int row_stride) {
    using T2 = RT::dtype;
    using U2 = base_types::packing<U>::packed_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size + (laneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + 2*(laneid % 4);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row+0)*row_stride + (col+0)]));
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row+0)*row_stride + (col+8)]));
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + 2*(laneid % 4);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row+8)*row_stride + (col+0)]));
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[(row+8)*row_stride + (col+8)]));
        }
    }
}
template<ducks::rt::col_layout RT, typename U>
__device__ inline static void load(RT &dst, const U *src, const int row_stride) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src[(row+0)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src[(row+0)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src[(row+1)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src[(row+1)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src[(row+8)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src[(row+8)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (laneid / 4);
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src[(row+9)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src[(row+9)*row_stride + (col+8)]);
        }
    }
}


template<ducks::rt::row_layout RT, typename U>
__device__ inline static void store(U *dst, const RT &src, const int row_stride) {
    using T2 = RT::dtype;
    using U2 = base_types::packing<U>::packed_type;
    int laneid = threadIdx.x % 32; 
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size + (laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + 2*(laneid % 4);
            *(U2*)(&dst[(row+0)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            *(U2*)(&dst[(row+0)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + 2*(laneid % 4);
            *(U2*)(&dst[(row+8)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            *(U2*)(&dst[(row+8)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}
// template<ducks::rt::row_layout RT, typename U>
// __device__ inline static void store(U *dst, const RT &src, const int row_stride) {
//     using T2 = RT::dtype;
//     using U2 = base_types::packing<U>::packed_type;
//     int laneid = kittens::laneid();
//     int warphalf = (laneid & 16) > 0;
//     int warphalflaneid = laneid % 16;
//     #pragma unroll
//     for(int i = 0; i < src.height; i++) {
//         int row_0to3 = i*src.tile_size + (warphalflaneid / 4);
//         int row = i*src.tile_size + (laneid / 4);
//         #pragma unroll
//         for(int j = 0; j < src.width; j++) {
//             int col = j*src.tile_size + warphalf*8 + 2*(laneid % 4);
//             U2 transfers[2];
//             transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
//             transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
//             transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
//             *(U2*)(&dst[(row_0to3+0)*row_stride + col]) = transfers[0];
//             *(U2*)(&dst[(row_0to3+4)*row_stride + col]) = transfers[1];
//         }
//         #pragma unroll
//         for(int j = 0; j < src.width; j++) {
//             int col = j*src.tile_size + warphalf*8 + 2*(laneid % 4);
//             U2 transfers[2];
//             transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
//             transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
//             transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
//             *(U2*)(&dst[(row_0to3+ 8)*row_stride + col]) = transfers[0];
//             *(U2*)(&dst[(row_0to3+12)*row_stride + col]) = transfers[1];
//         }
//     }
// }
template<ducks::rt::col_layout RT, typename U>
__device__ inline static void store(U *dst, const RT &src, const int row_stride) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (laneid / 4);
            dst[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}

// Loading a vector
template<typename RT, typename U>
__device__ inline static void load(RT &dst, const U *src) {
    using T2 = RT::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    
    int laneid = threadIdx.x % 32;
    auto row = 2*(laneid % 4);
    auto row_thread_id = laneid / 4; 
    
    if (dst.inner_dim == 2) {
        // applies for row_vec for row_layout or col_vec for col_layout 
        if (row_thread_id < 1) { 
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) { 
                int col = w*TILE_DIM;
                dst[w][row_thread_id].x = base_types::convertor<T, U>::convert(src[col + row + 0]);
                dst[w][row_thread_id].y = base_types::convertor<T, U>::convert(src[col + row + 1]); 
                dst[w][row_thread_id+1].x = base_types::convertor<T, U>::convert(src[col + row + 8]); 
                dst[w][row_thread_id+1].y = base_types::convertor<T, U>::convert(src[col + row + 9]); 
            }
        }
    }  else {
        // TODO: implement
    }
}


// Storing a vector
template<typename RT, typename U>
__device__ inline static void store(U *dst, const RT &src) {
    using T2 = RT::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<typename RT::dtype>::unpacked_type;

    int laneid = threadIdx.x % 32;
    auto row = 2*(laneid % 4);
    auto row_thread_id = laneid / 4;

    if (src.inner_dim == 2) {
        if(row_thread_id < 1) {
            #pragma unroll 
            for(auto w = 0; w < src.outer_dim; w++) { 
                int col = w*TILE_DIM;
                dst[col + row + 0] = base_types::convertor<U, T>::convert(src[w][row_thread_id].x);
                dst[col + row + 1] = base_types::convertor<U, T>::convert(src[w][row_thread_id].y);
                dst[col + row + 8] = base_types::convertor<U, T>::convert(src[w][row_thread_id+1].x);
                dst[col + row + 9] = base_types::convertor<U, T>::convert(src[w][row_thread_id+1].y);
            }
        }
    } else {
        // TODO: implement
    }
}


}