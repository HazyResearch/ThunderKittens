#pragma once

#include <concepts>

namespace kittens {

// row layouts are very much the default
struct st_naive_row_layout{}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
struct st_tma_row_layout{}; // only defined for x1, x2, x4 tiles.
struct st_xor_row_layout{}; // generic, non-tma swizzling mode

struct st_wgmma_row_0b_layout{ static constexpr int swizzling_mode=0; };
struct st_wgmma_row_32b_layout{ static constexpr int swizzling_mode=3; };
// however, we do need a few column layouts for wgmma mma's.
struct st_wgmma_col_t_0b_layout{ static constexpr int swizzling_mode=0; };
struct st_wgmma_col_t_32b_layout{ static constexpr int swizzling_mode=3; };

template<typename T>
concept st_type_2d_tma_layout = (
    std::is_same_v<T, st_naive_row_layout>   ||
    std::is_same_v<T, st_tma_row_layout> 
);
template<typename T>
concept st_wgmma_row_layout = (
    std::is_same_v<T, st_wgmma_row_0b_layout>   ||
    std::is_same_v<T, st_wgmma_row_32b_layout> 
);
template<typename T>
concept st_wgmma_col_layout = (
    std::is_same_v<T, st_wgmma_col_t_0b_layout>   // ||
    // std::is_same_v<T, st_wgmma_col_t_32b_layout> 
);
template<typename T>
concept st_row_layout = (
    st_wgmma_row_layout<T>                 ||
    std::is_same_v<T, st_naive_row_layout> ||
    std::is_same_v<T, st_tma_row_layout>   ||
    std::is_same_v<T, st_xor_row_layout>  
);
template<typename T>
concept st_col_layout = st_wgmma_col_layout<T> || std::is_same_v<T, st_wgmma_col_t_32b_layout>;
template<typename T>
concept st_wgmma_layout = st_wgmma_row_layout<T> || st_wgmma_col_layout<T>;
template<typename T>
concept st_layout = st_row_layout<T> || st_col_layout<T>;

namespace detail {

template<int height, int width, st_layout T=st_naive_row_layout>
struct shared_indexer {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return r*cols + c;
    }
};
template<int height, int width>
struct shared_indexer<height, width, st_tma_row_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return (r*cols + c) ^ (((r%8)*width/4)*8);
    }
};
template<int height, int width>
struct shared_indexer<height, width, st_xor_row_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int swizzling_bytes = sizeof(float4);
    static constexpr int swizzling_elements = swizzling_bytes / sizeof(bf16);
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return (r*cols + c) ^ (swizzling_elements*r);
    }
};
template<int height, int width>
struct shared_indexer<height, width, st_wgmma_row_0b_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    static constexpr int cols_per_idx1 = cols_per_core_matrix*2; // 16
    static constexpr int rows_per_idx2 = rows_per_core_matrix; // 8
    static constexpr int cols_per_idx3 = cols_per_core_matrix; // 8
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        int idx1 = c/cols_per_idx1;
        int idx2 = r/rows_per_idx2;
        int idx3 = (c%cols_per_idx1)/cols_per_idx3; // 4 is a constant specific to 32B swizzling
        int idx4 = (r%rows_per_core_matrix);
        int idx5 = (c%cols_per_core_matrix);
        return (
            (((idx1 * (2*height) // height is in units of 16, but we want units of 8
                + idx2) * 2 // * 2 tensormaps across
                + idx3) * 8 // * 8 rows per tensormap
                + idx4) * 8 // * 8 columns per row
                + idx5
        );
    }
};
template<int height, int width>
struct shared_indexer<height, width, st_wgmma_row_32b_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    static constexpr int cols_per_idx1 = cols_per_core_matrix*2; // 16
    static constexpr int rows_per_idx2 = rows_per_core_matrix; // 8
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        int idx1 = c/cols_per_idx1;
        int idx2 = r/rows_per_idx2;
        int idx3 = (r%rows_per_idx2)/4; // 4 is a constant specific to 32B swizzling
        int idx4 = (r%4)*2 + (c%cols_per_idx1)/8;
        int idx5 = (c%8);
        return (
            (((idx1 * (2*height) // height is in units of 16, but we want units of 8
                + idx2) * 2 // * 2 tensormaps across
                + idx3) * 8 // * 8 rows per tensormap
                + idx4) * 8 // * 8 columns per row
                + idx5
        ) ^ (idx3*8);
    }
};
// column layouts for wgmma
template<int height, int width>
struct shared_indexer<height, width, st_wgmma_col_t_0b_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    static constexpr int rows_per_idx1 = rows_per_core_matrix*2; // 16
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        int idx1 = (r/rows_per_idx1);
        int idx2 = c/cols_per_core_matrix;
        int idx3 = (r%rows_per_idx1)/rows_per_core_matrix;
        int idx4 = r%cols_per_core_matrix;
        int idx5 = c%rows_per_core_matrix;
        return (
            (((idx1 * (2*width) // height is in units of 16, but we want units of 8
             + idx2) * 2 // * 2 tensormaps across
             + idx3) * 8 // * 8 rows per tensormap
             + idx4) * 8 // * 8 columns per row
             + idx5
        );
    }
};
template<int height, int width>
struct shared_indexer<height, width, st_wgmma_col_t_32b_layout> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    static constexpr int rows_per_idx1 = rows_per_core_matrix*2; // 16
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        int idx1 = (r/rows_per_idx1);
        int idx2 = c/cols_per_core_matrix;
        int idx3 = (r%rows_per_core_matrix)/4;
        int idx4 = (r%4)*2 + (r%rows_per_idx1)/rows_per_core_matrix;
        int idx5 = c%cols_per_core_matrix;
        return (
            (((idx1 * (2*width) // height is in units of 16, but we want units of 8
             + idx2) * 2 // * 2 tensormaps across
             + idx3) * 8 // * 8 rows per tensormap
             + idx4) * 8 // * 8 columns per row
             + idx5
        ) ^ (idx3*8);
    }
};

}
}