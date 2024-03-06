#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 */
template<typename op, typename T>
__device__ static inline void unary_map(T &dst, const T &src) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param param Scalar parameter for the binary operation.
 */
template<typename op, typename T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with an unpacked scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param param Unpacked scalar parameter for the binary operation.
 */
template<typename op, typename T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the operation.
 * @param rhs Right-hand side source tile for the operation.
 */
template<typename op, typename T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

/* ----------  Row tile maps  ----------*/

/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param row_values Column vector containing values to apply across each row.
 */
// row-major layout
template<typename op, typename T, typename V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(is_rt_type<T>::value && T::layout == row_major, "T must be a valid tile type with row-major layout within the rt structure.");
    static_assert(is_rt_col_vec_type<V>::value && V::layout == col_major && V::outer_dim == T::height, "V must be a valid column vector type within the rt structure with size compatible to T.");

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}

/**
 * @brief Applies an operation across the rows of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param row_values Column vector containing values to apply across each row.
 */
// col-major layout
template<typename op, typename T, typename V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(is_rt_type<T>::value && T::layout == col_major, "T must be a valid tile type with column-major layout within the rt structure.");
    static_assert(is_rt_col_vec_type<V>::value && V::layout == col_major && V::outer_dim == T::height, "V must be a valid column vector type within the rt structure with size compatible to T.");

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}

/* ----------  Col major tile maps  ----------*/

/**
 * @brief Applies an operation across the columns of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param col_values Row vector containing values to apply across each column.
 */
// row-major layout
template<typename op, typename T, typename V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(is_rt_type<T>::value && T::layout == row_major, "T must be a valid tile type with row-major layout within the rt structure.");
    static_assert(is_rt_row_vec_type<V>::value && V::layout == row_major && V::outer_dim == T::width, "V must be a valid row vector type within the rt structure with size compatible to T.");

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the operation on.
 * @param col_values Row vector containing values to apply across each column.
 */
// col-major layout
template<typename op, typename T, typename V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(is_rt_type<T>::value && T::layout == col_major, "T must be a valid tile type with column-major layout within the rt structure.");
    static_assert(is_rt_row_vec_type<V>::value && V::layout == row_major && V::outer_dim == T::width, "V must be a valid row vector type within the rt structure with size compatible to T.");

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x); //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 */
template<typename T>
__device__ static inline void zero(T &dst) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::zero, T>(dst, dst);
}

/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 */
template<typename T>
__device__ static inline void one(T &dst) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::one, T>(dst, dst);
}

/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 */
template<typename T>
__device__ static inline void pos_infty(T &dst) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::pos_infty, T>(dst, dst);
}

/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 */
template<typename T>
__device__ static inline void neg_infty(T &dst) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the exponential function on.
 */
template<typename T>
__device__ static inline void exp(T &dst, const T &src) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::exp, T>(dst, src);
}

/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the absolute value function on.
 */
template<typename T>
__device__ static inline void abs(T &dst, const T &src) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::abs, T>(dst, src);
}

/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the ReLU function on.
 */
template<typename T>
__device__ static inline void relu(T &dst, const T &src) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    unary_map<base_ops::relu, T>(dst, src);
}

/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to copy from.
 */
template<typename T, typename U>
__device__ static inline void copy2(T &dst, const U &src) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps
/**
 * @brief Applies the max operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the operation.
 * @param rhs Right-hand side source tile or scalar for the operation.
 */
template<typename T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}

/**
 * @brief Applies the min operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the operation.
 * @param rhs Right-hand side source tile or scalar for the operation.
 */
template<typename T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}

/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the addition.
 * @param rhs Right-hand side source tile or scalar for the addition.
 */
template<typename T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}

/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the subtraction.
 * @param rhs Right-hand side source tile or scalar for the subtraction.
 */
template<typename T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}

/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the multiplication.
 * @param rhs Right-hand side source tile or scalar for the multiplication.
 */
template<typename T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}

/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst Destination tile where the result is stored.
 * @param lhs Left-hand side source tile for the division.
 * @param rhs Right-hand side source tile or scalar for the division.
 */
template<typename T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

// row maps
/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the addition on.
 * @param row_values Column vector containing values to add to each row.
 */
template<typename T, typename V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}

/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the subtraction on.
 * @param row_values Column vector containing values to subtract from each row.
 */
template<typename T, typename V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}

/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the multiplication on.
 * @param row_values Column vector containing values to multiply each row by.
 */
template<typename T, typename V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}

/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the division on.
 * @param row_values Column vector containing values to divide each row by.
 */
template<typename T, typename V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    row_map<base_ops::div, T, V>(dst, src, row_values);
}

// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the addition on.
 * @param col_values Row vector containing values to add to each column.
 */
template<typename T, typename V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}

/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the subtraction on.
 * @param col_values Row vector containing values to subtract from each column.
 */
template<typename T, typename V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}

/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the multiplication on.
 * @param col_values Row vector containing values to multiply each column by.
 */
template<typename T, typename V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}

/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst Destination tile where the result is stored.
 * @param src Source tile to apply the division on.
 * @param col_values Row vector containing values to divide each column by.
 */
template<typename T, typename V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    static_assert(is_rt_type<T>::value, "T must be a valid tile type within the rt structure.");
    col_map<base_ops::div, T, V>(dst, src, col_values);
}

}
