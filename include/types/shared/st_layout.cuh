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

/**
 * @brief A naive row-major layout with no swizzling. row*(#cols)+c
 */
struct naive {}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.
/**
 * @brief A layout for minimal bank conflicts and maximal coalescing.
 *
 */
struct swizzle {}; // only defined for x1, x2, x4 tiles.

/**
 * @brief A layout specialized to match both TMA and WGMMA.
 *
 * Note this layout has worse coalescing than the standard swizzle mode
 * for tiles that are a have width that isn't a multiple of 4,
 * unless the width is exactly 1 or 2.
 */
struct wgmma_swizzle {}; // only defined for x1, x2, x4 tiles.
/**
 * @brief A layout for wgmma with no swizzling. This mode is necessary for the wgmma transpose.
 * 
 * Note, it has worse coalescing and bank conflicts than any other mode.
 */
struct wgmma_interleave { static constexpr int swizzling_mode=0; };

/**
 * @brief Concept to check if a type is a row-contiguous layout.
 */
template<typename T>
concept all = (
    std::is_same_v<T, naive>            ||
    std::is_same_v<T, swizzle>          ||
    std::is_same_v<T, wgmma_swizzle>    ||
    std::is_same_v<T, wgmma_interleave>
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
template<typename T, int height, int width, ducks::st_layout::all L=ducks::st_layout::naive>
struct shared_indexer {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    /**
     * @brief Get a memory offset from a row and column index.
     */
    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major index default
        return &ptr[r*cols + c];
    }
};
template<typename T, int height, int width>
struct shared_indexer<T, height, width, ducks::st_layout::swizzle> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int swizzle_repeat = sizeof(T) == 2 ? ((width%4==0) ? 1024 : (width%2==0) ? 512 : 256)
                                        : sizeof(T) == 4 ? ((width%2==0) ? 1024 : 512) : -1;
    static constexpr int swizzle_shift  = sizeof(T) == 2 ? ((width%4==0) ? 6 : (width%2==0) ? 5 : 4)
                                        : sizeof(T) == 4 ? ((width%2==0) ? 6 : 5) : -1;
    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major index default
        const uint64_t addr = (uint64_t)(&ptr[r*cols + c]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    }
};
template<typename T, int height, int width>
struct shared_indexer<T, height, width, ducks::st_layout::wgmma_swizzle> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int swizzle_repeat = sizeof(T) == 2 ? ((width%4==0) ? 1024 : (width%2==0) ? 512 : 256)
                                        : sizeof(T) == 4 ? ((width%2==0) ? 1024 : 512) : -1;
    static constexpr int swizzle_shift  = sizeof(T) == 2 ? ((width%4==0) ? 6 : (width%2==0) ? 5 : 4)
                                        : sizeof(T) == 4 ? ((width%2==0) ? 6 : 5) : -1;
    static constexpr int subtile_cols   = sizeof(T) == 2 ? ((width%4==0) ? 64 : (width%2==0) ? 32 : 16)
                                        : sizeof(T) == 4 ? ((width%2==0) ? 32 : 16) : -1;
    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major index default
        const int outer_idx = c/subtile_cols;
        const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    }
};
template<typename T, int height, int width>
struct shared_indexer<T, height, width, ducks::st_layout::wgmma_interleave> {
    static constexpr int rows = height*16;
    static constexpr int cols = width*16;
    static constexpr int rows_per_core_matrix = 8;
    static constexpr int cols_per_core_matrix = 16 / sizeof(T);
    __device__ static inline T* idx(T *ptr, int r, int c) { // naive row-major index default
        int idx1 = r/rows_per_core_matrix;
        int idx2 = c/cols_per_core_matrix;
        int idx3 = (r%rows_per_core_matrix);
        int idx4 = (c%cols_per_core_matrix);
        return &ptr[(
            (
                (
                    idx1 * ((kittens::TILE_DIM/cols_per_core_matrix)*width) // width is in units of 16, but we want units of core matrices
                    + idx2
                ) * rows_per_core_matrix
                + idx3
            ) * cols_per_core_matrix
            + idx4
        )];
    }
};

}
}
