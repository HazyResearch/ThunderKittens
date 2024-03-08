#pragma once

#include <concepts>

namespace kittens {

// row layouts are very much the default
struct st_naive_row_layout{}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
struct st_xor_row_layout{}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
struct st_wgmma_row_0b_layout{ static constexpr int swizzling_mode=0; };
struct st_wgmma_row_32b_layout{ static constexpr int swizzling_mode=3; };
struct st_wgmma_row_64b_layout{ static constexpr int swizzling_mode=2; };
struct st_wgmma_row_128b_layout{ static constexpr int swizzling_mode=1; };
// however, we do need a few column layouts for wgmma mma's.
struct st_wgmma_col_0b_layout{ static constexpr int swizzling_mode=0; };
struct st_wgmma_col_32b_layout{ static constexpr int swizzling_mode=3; };
struct st_wgmma_col_64b_layout{ static constexpr int swizzling_mode=2; };
struct st_wgmma_col_128b_layout{ static constexpr int swizzling_mode=1; };

struct st_wgmma_col_t_0b_layout{ static constexpr int swizzling_mode=0; };
struct st_wgmma_col_t_32b_layout{ static constexpr int swizzling_mode=3; };
struct st_wgmma_col_t_64b_layout{ static constexpr int swizzling_mode=2; };
struct st_wgmma_col_t_128b_layout{ static constexpr int swizzling_mode=1; };

template<typename T>
concept st_wgmma_row_layout = (
    std::is_same_v<T, st_wgmma_row_0b_layout>   ||
    std::is_same_v<T, st_wgmma_row_32b_layout>  ||
    std::is_same_v<T, st_wgmma_row_64b_layout>  ||
    std::is_same_v<T, st_wgmma_row_128b_layout>
);
template<typename T>
concept st_wgmma_col_layout = (
    std::is_same_v<T, st_wgmma_col_0b_layout>     ||
    std::is_same_v<T, st_wgmma_col_32b_layout>    ||
    std::is_same_v<T, st_wgmma_col_64b_layout>    ||
    std::is_same_v<T, st_wgmma_col_128b_layout>   || 
    std::is_same_v<T, st_wgmma_col_t_0b_layout>   ||
    std::is_same_v<T, st_wgmma_col_t_32b_layout>  ||
    std::is_same_v<T, st_wgmma_col_t_64b_layout>  ||
    std::is_same_v<T, st_wgmma_col_t_128b_layout>
);
template<typename T>
concept st_row_layout = (
    st_wgmma_row_layout<T>                  ||
    std::is_same_v<T, st_naive_row_layout>  ||
    std::is_same_v<T, st_xor_row_layout>
);
template<typename T>
concept st_col_layout = st_wgmma_col_layout<T>; // just an alias for now
template<typename T>
concept st_wgmma_layout = st_wgmma_row_layout<T> || st_wgmma_col_layout<T>;
template<typename T>
concept st_layout = st_row_layout<T> || st_col_layout<T>;

namespace detail {

template<st_layout T>
__device__ inline int st_idx(int r, int c, int height, int width);

template<>
__device__ inline int st_idx<st_naive_row_layout>(int r, int c, int height, int width) {
    return r*width*16 + c;
}
template<>
__device__ inline int st_idx<st_xor_row_layout>(int r, int c, int height, int width) { // xor's in 8-element chunks (16 bytes for bf16)
    return (r*width*16 + c) ^ (8*r);
}
template<>
__device__ inline int st_idx<st_wgmma_row_0b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height) // height is in units of 16, but we want units of 8
            + (r/8)) * 2 // * 2 tensormaps across
            + ((c%16)/8)) * 8 // * 8 rows per tensormap
            + (r%8)) * 8 // * 8 columns per row
            + (c%8)
    );
}

template<>
__device__ inline int st_idx<st_wgmma_row_32b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height) // height is in units of 16, but we want units of 8
            + (r/8)) * 2 // * 2 tensormaps across
            + ((r%8)/4)) * 8 // * 8 rows per tensormap
            + (r%4)*2 + (c%16)/8) * 8 // * 8 columns per row
            + (c%8)
    ) ^ (((r%8)/4)*8);
}
template<>
__device__ inline int st_idx<st_wgmma_row_64b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height) // height is in units of 16, but we want units of 8
            + (r/16)*2 + (r%8)/4) * 2 // * 2 tensormaps across
            + ((r%4)/2)) * 8 // * 8 rows per tensormap
            + (r%2)*4 + ((r%16)/8)*2 + (c%16)/8) * 8 // * 8 columns per row
            + (c%8)
    ) ^ (((r%8)/4)*16 + ((r%4)/2)*8);
}
template<>
__device__ inline int st_idx<st_wgmma_row_128b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height) // height is in units of 16, but we want units of 8
            + (r/32)*4 + (r%8)/2) * 2 // * 2 tensormaps across
            + (r%2)) * 8 // * 8 rows per tensormap
            + ((r%32)/8)*2 + (c%16)/8) * 8 // * 8 columns per row
            + (c%8)
    ) ^ (((r%8)/2)*16 + (r%2)*8);
}
// column layouts for wgmma
template<>
__device__ inline int st_idx<st_wgmma_col_0b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width) // height is in units of 16, but we want units of 8
            + (c/8)) * 2 // * 2 tensormaps across
            + ((r%16)/8)) * 8 // * 8 rows per tensormap
            + (c%8)) * 8 // * 8 columns per row
            + (r%8)
    );
}
template<>
__device__ inline int st_idx<st_wgmma_col_t_0b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width) // height is in units of 16, but we want units of 8
            + (c/8)) * 2 // * 2 tensormaps across
            + ((r%16)/8)) * 8 // * 8 rows per tensormap
            + (r%8)) * 8 // * 8 columns per row
            + (c%8)
    );
}
template<>
__device__ inline int st_idx<st_wgmma_col_32b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width) // height is in units of 16, but we want units of 8
            + (c/8)) * 2 // * 2 tensormaps across
            + ((c%8)/4)) * 8 // * 8 rows per tensormap
            + (c%4)*2 + (r%16)/8) * 8 // * 8 columns per row
            + (r%8)
    ) ^ (((c%8)/4)*8);
}
template<>
__device__ inline int st_idx<st_wgmma_col_64b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width) // height is in units of 16, but we want units of 8
            + (c/16)*2 + (c%8)/4) * 2 // * 2 tensormaps across
            + ((c%4)/2)) * 8 // * 8 rows per tensormap
            + (c%2)*4 + ((c%16)/8)*2 + (r%16)/8) * 8 // * 8 columns per row
            + (r%8)
    ) ^ (((c%8)/4)*16 + ((c%4)/2)*8);
}
template<>
__device__ inline int st_idx<st_wgmma_col_128b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width) // height is in units of 16, but we want units of 8
            + (c/32)*4 + (c%8)/2) * 2 // * 2 tensormaps across
            + (c%2)) * 8 // * 8 rows per tensormap
            + ((c%32)/8)*2 + (r%16)/8) * 8 // * 8 columns per row
            + (r%8)
    ) ^ (((c%8)/2)*16 + (c%2)*8);
}

}

}