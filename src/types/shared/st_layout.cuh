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
struct xor_swizzle {}; // only defined for x1, x2, x4 tiles.

/**
 * @brief A row layout for wgmma with no swizzling.
 */
struct interleave { static constexpr int swizzling_mode=0; };

/**
 * @brief Concept to check if a type is a wgmma row layout.
 */
template<typename T>
concept wgmma_normal = (
    std::is_same_v<T, interleave>   ||
    std::is_same_v<T, xor_swizzle> 
);
/**
 * @brief Concept to check if a type is a wgmma column layout.
 */
template<typename T>
concept wgmma_transposed = (
    std::is_same_v<T, interleave>   // ||
);
/**
 * @brief Concept to check if a type is a row-contiguous layout.
 */
template<typename T>
concept all = (
    std::is_same_v<T, naive>             ||
    std::is_same_v<T, xor_swizzle>       ||
    std::is_same_v<T, interleave>
);

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
    /**
     * @brief Get a memory offset from a row and column index.
     */
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return r*cols + c;
    }
};
template<int height, int width>
struct shared_indexer<height, width, ducks::st_layout::xor_swizzle> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        return (r*cols + c) ^ (((r%8)*width/4)*8);
    }
};
template<int height, int width>
struct shared_indexer<height, width, ducks::st_layout::interleave> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 8;
    __device__ static inline int idx(int r, int c) { // naive row-major index default
        int idx1 = r/rows_per_core_matrix;
        int idx2 = c/cols_per_core_matrix;
        int idx3 = (r%rows_per_core_matrix);
        int idx4 = (c%cols_per_core_matrix);
        return (
            (
                (
                    idx1 * (2*width) // width is in units of 16, but we want units of 8
                    + idx2
                ) * 8 // * 8 rows per tensormap
                + idx3
            ) * 8 // * 8 columns per row
            + idx4
        );
    }
};

}
}
