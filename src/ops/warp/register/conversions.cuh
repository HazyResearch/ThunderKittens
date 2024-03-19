#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

__device__ inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    asm volatile (
        "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
    :   "+r"(*(uint32_t*)(&dst))
    :   "r"(*(uint32_t*)(&src))
    );
}
template<typename T2, ducks::rt_layout::all layout>
__device__ inline void swap_layout(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T2, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _height, _width, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

template<typename T2, ducks::rt_layout::all layout>
__device__ inline void transpose(rt_base<T2, layout> &dst, const rt_base<T2, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void transpose_sep(rt<T2, _width, _height, layout> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < _height; i++) {
        #pragma unroll
        for(int j = 0; j < _width; j++) {
            transpose(dst.tiles[j][i], src.tiles[i][j]);
        }
    }
}

template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _height, _width, layout>& transpose_inplace(rt<T2, _height, _width, layout> &tile) {
    static_assert(_width == _height, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < _height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

template<typename T2, typename U2, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T2, layout> &dst, const rt_base<U2, layout> &src) {
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  SUBTILE  ---------- */

template<ducks::rt::all RT, int subtile_height>
__device__ inline rt<typename RT::dtype, subtile_height, RT::width, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    static_assert(RT::height % subtile_height == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::dtype, subtile_height, RT::width, typename RT::layout>&>(
        src.tiles[idx*subtile_height]
    );
}

}