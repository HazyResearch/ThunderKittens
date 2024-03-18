#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
namespace st_layout {

// row layouts are very much the default
struct naive {}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
struct tma_swizzle {}; // only defined for x1, x2, x4 tiles.
struct xor_swizzle {}; // generic, non-tma swizzling mode

struct wgmma_row_0b { static constexpr int swizzling_mode=0; };
struct wgmma_row_32b { static constexpr int swizzling_mode=3; };
// however, we do need a few column layouts for wgmma mma's.
struct wgmma_col_t_0b{ static constexpr int swizzling_mode=0; };
struct wgmma_col_t_32b{ static constexpr int swizzling_mode=3; }; // Swizzled transposed layout not yet working

template<typename T>
concept tma_2d = (
    std::is_same_v<T, naive>   ||
    std::is_same_v<T, tma_swizzle> 
);
template<typename T>
concept wgmma_row = (
    std::is_same_v<T, wgmma_row_0b>   ||
    std::is_same_v<T, wgmma_row_32b> 
);
template<typename T>
concept wgmma_col = (
    std::is_same_v<T, wgmma_col_t_0b>   // ||
    // std::is_same_v<T, wgmma_col_t_32b> 
);
template<typename T>
concept row = (
    wgmma_row<T>  ||
    wgmma_col<T>  || // wgmma col_t layouts are actually row layouts in terms of local contiguity.
    tma_2d<T>     ||
    std::is_same_v<T, xor_swizzle>  ||
    std::is_same_v<T, wgmma_col_t_32b>   // temporary, until it merges into wgmma_col
);
template<typename T>
concept col = false; // There are actually no column layouts right now. Which is good because they're slow!
template<typename T>
concept wgmma = wgmma_row<T> || wgmma_col<T>;
template<typename T>
concept all = row<T> || col<T>;

}
} // namespace ducks

namespace detail {

template<int height, int width, ducks::st_layout::all T=ducks::st_layout::naive>
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
struct shared_indexer<height, width, ducks::st_layout::tma_swizzle> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return (r*cols + c) ^ (((r%8)*width/4)*8);
    }
};
template<int height, int width>
struct shared_indexer<height, width, ducks::st_layout::xor_swizzle> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int swizzling_bytes = sizeof(float4);
    static constexpr int swizzling_elements = swizzling_bytes / sizeof(bf16);
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return (r*cols + c) ^ (swizzling_elements*r);
    }
};
template<int height, int width>
struct shared_indexer<height, width, ducks::st_layout::wgmma_row_0b> {
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
struct shared_indexer<height, width, ducks::st_layout::wgmma_row_32b> {
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
struct shared_indexer<height, width, ducks::st_layout::wgmma_col_t_0b> {
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
// This is what it seems like it should be but the hardware appears to disagree.
template<int height, int width>
struct shared_indexer<height, width, ducks::st_layout::wgmma_col_t_32b> {
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
