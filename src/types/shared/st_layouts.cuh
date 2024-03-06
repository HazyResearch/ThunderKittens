#pragma once

namespace kittens {

/**
 * @brief A naive row layout with no swizzling.
 */
struct st_naive_row_layout{}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.

/**
 * @brief A row layout with XOR swizzling in 8-element chunks (16 bytes for bf16).
 */
struct st_xor_row_layout{}; // swizzling_mode left undefined to cause errors if matrix_descriptor is called.

/**
 * @brief A row layout for wgmma with no swizzling.
 */
struct st_wgmma_row_0b_layout{ static constexpr int swizzling_mode=0; };

/**
 * @brief A row layout for wgmma with 32-bit swizzling.
 */
struct st_wgmma_row_32b_layout{ static constexpr int swizzling_mode=3; };

/**
 * @brief A row layout for wgmma with 64-bit swizzling.
 */
struct st_wgmma_row_64b_layout{ static constexpr int swizzling_mode=2; };

/**
 * @brief A row layout for wgmma with 128-bit swizzling.
 */
struct st_wgmma_row_128b_layout{ static constexpr int swizzling_mode=1; };

/**
 * @brief A column layout for wgmma with no swizzling.
 */
struct st_wgmma_col_0b_layout{ static constexpr int swizzling_mode=0; };

/**
 * @brief A column layout for wgmma with 32-bit swizzling.
 */
struct st_wgmma_col_32b_layout{ static constexpr int swizzling_mode=3; };

/**
 * @brief A column layout for wgmma with 64-bit swizzling.
 */
struct st_wgmma_col_64b_layout{ static constexpr int swizzling_mode=2; };

/**
 * @brief A column layout for wgmma with 128-bit swizzling.
 */
struct st_wgmma_col_128b_layout{ static constexpr int swizzling_mode=1; };

/**
 * @brief Struct to check if a type is a wgmma row layout.
 */
template<typename T>
struct st_wgmma_row_layout : std::integral_constant<bool,
    std::is_same<T, st_wgmma_row_0b_layout>::value   ||
    std::is_same<T, st_wgmma_row_32b_layout>::value  ||
    std::is_same<T, st_wgmma_row_64b_layout>::value  ||
    std::is_same<T, st_wgmma_row_128b_layout>::value
> {};

/**
 * @brief Struct to check if a type is a wgmma column layout.
 */
template<typename T>
struct st_wgmma_col_layout : std::integral_constant<bool,
    std::is_same<T, st_wgmma_col_0b_layout>::value   ||
    std::is_same<T, st_wgmma_col_32b_layout>::value  ||
    std::is_same<T, st_wgmma_col_64b_layout>::value  ||
    std::is_same<T, st_wgmma_col_128b_layout>::value
> {};

/**
 * @brief Struct to check if a type is a row layout.
 */
template<typename T>
struct st_row_layout : std::integral_constant<bool,
    st_wgmma_row_layout<T>::value                  ||
    std::is_same<T, st_naive_row_layout>::value    ||
    std::is_same<T, st_xor_row_layout>::value
> {};

/**
 * @brief Struct to check if a type is a column layout.
 */
template<typename T>
struct st_col_layout : std::integral_constant<bool,
    st_wgmma_col_layout<T>::value
> {}; // just an alias for now

/**
 * @brief Struct to check if a type is a wgmma layout.
 */
template<typename T>
struct st_wgmma_layout : std::integral_constant<bool,
    st_wgmma_row_layout<T>::value || st_wgmma_col_layout<T>::value
> {};

/**
 * @brief Struct to check if a type is any layout.
 */
template<typename T>
struct st_layout : std::integral_constant<bool,
    st_row_layout<T>::value || st_col_layout<T>::value
> {};

namespace detail {

/**
 * @brief Function template to calculate the index in a shared memory tile based on the layout.
 *
 * @tparam T The layout type.
 * @param r The row position.
 * @param c The column position.
 * @param height The height of the tile.
 * @param width The width of the tile.
 * @return The calculated index.
 */
template<typename T, std::enable_if_t<st_layout<T>::value, int> = 0>
__device__ inline int st_idx(int r, int c, int height, int width);

/**
 * @brief Specialization of st_idx for st_naive_row_layout.
 */
template<>
__device__ inline int st_idx<st_naive_row_layout>(int r, int c, int height, int width) {
    return r*width*16 + c;
}

/**
 * @brief Specialization of st_idx for st_xor_row_layout.
 */
template<>
__device__ inline int st_idx<st_xor_row_layout>(int r, int c, int height, int width) {
    return (r*width*16 + c) ^ (8*r);
}

/**
 * @brief Specialization of st_idx for st_wgmma_row_0b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_row_0b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height)
            + (r/8)) * 2
            + ((c%16)/8)) * 8
            + (r%8)) * 8
            + (c%8)
    );
}

/**
 * @brief Specialization of st_idx for st_wgmma_row_32b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_row_32b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height)
            + (r/8)) * 2
            + ((r%8)/4)) * 8
            + (r%4)*2 + (c%16)/8) * 8
            + (c%8)
    ) ^ (((r%8)/4)*8);
}

/**
 * @brief Specialization of st_idx for st_wgmma_row_64b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_row_64b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height)
            + (r/16)*2 + (r%8)/4) * 2
            + ((r%4)/2)) * 8
            + (r%2)*4 + ((r%16)/8)*2 + (c%16)/8) * 8
            + (c%8)
    ) ^ (((r%8)/4)*16 + ((r%4)/2)*8);
}

/**
 * @brief Specialization of st_idx for st_wgmma_row_128b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_row_128b_layout>(int r, int c, int height, int width) {
    return (
        ((((c/16) * (2*height)
            + (r/32)*4 + (r%8)/2) * 2
            + (r%2)) * 8
            + ((r%32)/8)*2 + (c%16)/8) * 8
            + (c%8)
    ) ^ (((r%8)/2)*16 + (r%2)*8);
}

/**
 * @brief Specialization of st_idx for st_wgmma_col_0b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_col_0b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width)
            + (c/8)) * 2
            + ((r%16)/8)) * 8
            + (c%8)) * 8
            + (r%8)
    );
}

/**
 * @brief Specialization of st_idx for st_wgmma_col_32b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_col_32b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width)
            + (c/8)) * 2
            + ((c%8)/4)) * 8
            + (c%4)*2 + (r%16)/8) * 8
            + (r%8)
    ) ^ (((c%8)/4)*8);
}

/**
 * @brief Specialization of st_idx for st_wgmma_col_64b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_col_64b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width)
            + (c/16)*2 + (c%8)/4) * 2
            + ((c%4)/2)) * 8
            + (c%2)*4 + ((c%16)/8)*2 + (r%16)/8) * 8
            + (r%8)
    ) ^ (((c%8)/4)*16 + ((c%4)/2)*8);
}

/**
 * @brief Specialization of st_idx for st_wgmma_col_128b_layout.
 */
template<>
__device__ inline int st_idx<st_wgmma_col_128b_layout>(int r, int c, int height, int width) {
    return (
        ((((r/16) * (2*width)
            + (c/32)*4 + (c%8)/2) * 2
            + (c%2)) * 8
            + ((c%32)/8)*2 + (r%16)/8) * 8
            + (r%8)
    ) ^ (((c%8)/2)*16 + (c%2)*8);
}

}

}
