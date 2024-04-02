/**
 * @file
 * @brief A collection of layouts and indexing patterns for shared memory tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace st_layout
 * 
 * @brief A namespace for template metaprogramming with shared tile layouts.
 */
namespace st_layout {

// row layouts are very much the default
/**
 * @brief A naive row layout with no swizzling.
 */
struct naive {}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
/**
 * @brief A row layout with swizzling specialized to match TMA modes.
 *
 * Note this is only defined for Nx1, Nx2, and Nx4 shared tiles.
 */
struct tma_swizzle {}; // only defined for x1, x2, x4 tiles.
/**
 * @brief A row layout with XOR swizzling in 8-element chunks (16 bytes for bf16).
 */
struct xor_swizzle {}; // generic, non-tma swizzling mode

/**
 * @brief A row layout for wgmma with no swizzling.
 */
struct wgmma_row_0b { static constexpr int swizzling_mode=0; };
/**
 * @brief A row layout for wgmma with 32-bit swizzling.
 */
struct wgmma_row_32b { static constexpr int swizzling_mode=3; };
/**
 * @brief A (transposed) column layout for wgmma with no swizzling.
 */
struct wgmma_col_t_0b{ static constexpr int swizzling_mode=0; };
/**
 * @brief A (transposed) column layout for wgmma with 32-bit swizzling.
 */
struct wgmma_col_t_32b{ static constexpr int swizzling_mode=3; }; // Swizzled transposed layout not yet working

/**
 * @brief Concept to check if a type is a wgmma row layout.
 */
template<typename T>
concept wgmma_row = (
    std::is_same_v<T, wgmma_row_0b>   ||
    std::is_same_v<T, wgmma_row_32b> 
);
/**
 * @brief Concept to check if a type is a wgmma column layout.
 */
template<typename T>
concept wgmma_col = (
    std::is_same_v<T, wgmma_col_t_0b>   // ||
    // std::is_same_v<T, wgmma_col_t_32b> -- this doesn't work right now.
);
/**
 * @brief Concept to check if a type is a row-contiguous layout.
 */
template<typename T>
concept row = (
    wgmma_row<T>  ||
    wgmma_col<T>  || // wgmma col_t layouts are actually row layouts in terms of local contiguity.
    tma_2d<T>     ||
    std::is_same_v<T, xor_swizzle>  ||
    std::is_same_v<T, wgmma_col_t_32b>   // temporary, until it merges into wgmma_col
);
/**
 * @brief Concept to check if a type is a col-contiguous layout.
 */
template<typename T>
concept col = false; // There are actually no column layouts right now. Which is good because they're slow!
/**
 * @brief Concept to check if a type is an st_layout
 */
template<typename T>
concept all = row<T> || col<T>;

}
} // namespace ducks

/**
 * @namespace detail
 *
 * @brief A namespace for internal calculations that really don't need to be exposed.
 */
namespace detail {

/**
 * @brief Struct template to calculate addresses in shared memory tiles
 *
 * @tparam height The tile height, in subtiles of 16.
 * @tparam width The tile width, in subtiles of 16.
 * @tparam T The layout type.
 * @param r[in] The row position.
 * @param c[in] The column position.
 * @return The calculated index.
 */
template<int height, int width, ducks::st_layout::all T=ducks::st_layout::naive>
struct shared_indexer {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    /**
     * @brief Get a memory offset from a row and column index.
     */
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
