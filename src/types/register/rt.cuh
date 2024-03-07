#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "../../common/base_types.cuh"

#include "rt_layouts.cuh"
#include "rt_base.cuh"

namespace kittens {

struct rt_id {}; // Unique identifier for the rt structure.
struct rt_col_vec_id {}; // Unique identifier for the col_vec structure.
struct rt_row_vec_id {}; // Unique identifier for the row_vec structure.

/**
 * @brief Main tile structure for matrix tiles handling.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the matrix tile, either row-major or column-major.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles. It supports both
 * row-major and column-major layouts and includes helper structs for type inference
 * in vector maps.
 */
template<packed_type T2, int _height, int _width, rt_layout _layout=rt_row_layout>
struct rt {
    using identifier = rt_id; ///< Type identifier for the rt structure.
    using layout = _layout; ///< Layout of the matrix tile.
    using dtype = T2; ///< Data type of the matrix elements.

    static constexpr int height              = _height; ///< Height in subtiles.
    static constexpr int width               = _width; ///< Width in subtiles.
    static constexpr int rows                = height  * rt_base<dtype, layout>::tile_size; ///< Total number of rows.
    static constexpr int cols                = width * rt_base<dtype, layout>::tile_size; ///< Total number of columns.
    static constexpr int tile_size           = rt_base<dtype, layout>::tile_size; ///< Size of the base tile.
    static constexpr int num_elements        = rt_base<dtype, layout>::num_elements        * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<dtype, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread   = rt_base<dtype, layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile     = rt_base<dtype, layout>::packed_per_thread; ///< Packed elements per tile.

    rt_base<dtype, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

    // Correcting the dependent type names with the typename keyword
    using col_type = typename std::conditional<layout::row, typename rt_base<dtype, layout>::col_type, typename rt_base<dtype, layout>::row_type>::type;
    using row_type = typename std::conditional<layout::row, typename rt_base<dtype, layout>::row_type, typename rt_base<dtype, layout>::col_type>::type;

    /**
     * @brief Helper struct for column vector type inference.
     *
     * This struct represents a column vector within the matrix tile, allowing for
     * operations specific to columns. It is used for type inference in vector maps.
     */
    struct col_vec {
        using identifier = rt_col_vec_id; ///< Type identifier for the col_vec structure.
        using layout = _layout; ///< Layout of the matrix tile.
        using dtype = T2; ///< Data type of the matrix elements.

        typename rt_base<dtype, layout>::col_type data[height]; ///< Storage for the column vector.

        static constexpr int outer_dim = height; ///< Number of elements in the outer dimension.
        static constexpr int inner_dim = sizeof(data[0])/sizeof(data[0][0]); ///< Number of elements in the inner dimension.

        __device__ inline       typename rt_base<dtype, layout>::col_type& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const typename rt_base<dtype, layout>::col_type& operator[](size_t idx) const { return data[idx]; }
    };

    /**
     * @brief Helper struct for row vector type inference.
     *
     * This struct represents a row vector within the matrix tile, allowing for
     * operations specific to rows. It is used for type inference in vector maps.
     */
    struct row_vec {
        using identifier = rt_row_vec_id; ///< Type identifier for the row_vec structure.
        using layout = _layout; ///< Layout of the matrix tile.
        using dtype = T2; ///< Data type of the matrix elements.

        typename rt_base<dtype, layout>::row_type data[width]; ///< Storage for the row vector.

        static constexpr int outer_dim = width; ///< Number of elements in the outer dimension.
        static constexpr int inner_dim = sizeof(data[0])/sizeof(data[0][0]); ///< Number of elements in the inner dimension.

        __device__ inline       typename rt_base<dtype, layout>::row_type& operator[](size_t idx)       { return data[idx]; }
        __device__ inline const typename rt_base<dtype, layout>::row_type& operator[](size_t idx) const { return data[idx]; }
    };
};

/* ----------  CONCEPTS  ---------- */

template<typename T> concept rt_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_id>; // Checks if T::dentifier is abstract_rt
// specialized types for specialized functions
template<typename T>
concept rt_type_rowlayout = rt_type<T> && std::is_same_v<typename T::layout, rt_row_layout>;
template<typename T>
concept rt_type_collayout = rt_type<T> && std::is_same_v<typename T::layout, rt_col_layout>;

template<typename T>
concept rt_col_vec_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_col_vec_id>; // Checks if T::identifier is abstract_vector
template<typename T>
concept rt_row_vec_type = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, rt_row_vec_id>; // Checks if T::identifier is abstract_vector
template<typename T>
concept rt_vec_type = rt_row_vec_type<T> || rt_col_vec_type<T>;



/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _height, int _width, rt_layout layout=rt_row_layout> using rt_fl = rt<float2, _height, _width, layout>;
template<int _height, int _width, rt_layout layout=rt_row_layout> using rt_bf = rt<bf16_2, _height, _width, layout>;

// layout, type, and size wrappers
// sizes are chosen with the assumption that you aren't going to want to fit more than
// 8 subtiles on a thread. (if you need more, you can of course add your own using's.)

template<rt_layout layout=rt_row_layout> using rt_fl_1x1 = rt_fl<1, 1, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x2 = rt_fl<1, 2, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x4 = rt_fl<1, 4, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_1x8 = rt_fl<1, 8, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x1 = rt_fl<2, 1, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x2 = rt_fl<2, 2, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_2x4 = rt_fl<2, 4, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_4x1 = rt_fl<4, 1, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_4x2 = rt_fl<4, 2, layout>; // 64 registers used
template<rt_layout layout=rt_row_layout> using rt_fl_8x1 = rt_fl<8, 1, layout>; // 64 registers used

template<rt_layout layout=rt_row_layout> using rt_bf_1x1 = rt_bf<1, 1, layout>; //  4 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x2 = rt_bf<1, 2, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x4 = rt_bf<1, 4, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_1x8 = rt_bf<1, 8, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x1 = rt_bf<2, 1, layout>; //  8 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x2 = rt_bf<2, 2, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_2x4 = rt_bf<2, 4, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_4x1 = rt_bf<4, 1, layout>; // 16 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_4x2 = rt_bf<4, 2, layout>; // 32 registers used
template<rt_layout layout=rt_row_layout> using rt_bf_8x1 = rt_bf<8, 1, layout>; // 32 registers used

}