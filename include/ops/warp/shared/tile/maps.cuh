/**
 * @file
 * @brief Warp-scope maps on shared tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Performs a uniform unary operation on a tile.
 * 
 * This function applies a given unary operation to each element of the source tile and stores the result in the destination tile.
 * The operation is applied independently to each element, without considering its position or the values of neighboring elements.
 * 
 * @tparam op The unary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the unary operation is applied.
 */
template<typename op, ducks::st::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i]);
    }
}

/**
 * @brief Performs a uniform binary operation on a tile with a scalar parameter.
 * 
 * This function applies a given binary operation to each element of the source tile and a scalar parameter, then stores the result in the destination tile.
 * The operation is applied independently to each element, treating the scalar parameter as the second operand for each operation.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the scalar parameter.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] param The scalar parameter to be used as the second operand in the binary operation.
 */
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i], param);
    }
}

/**
 * @brief Performs a uniform binary operation on two tiles.
 * 
 * This function applies a given binary operation to corresponding elements of two source tiles and stores the result in the destination tile.
 * The operation is applied independently to each pair of elements, without considering their positions or the values of neighboring elements.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile to which the binary operation is applied.
 * @param[in] rhs The second source tile to which the binary operation is applied.
 */
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst.data[i] = op::template op<typename T::dtype>(lhs.data[i], rhs.data[i]);
    }
}

/**
 * @brief Performs a row-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each row of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each row, using the vector element as 
 * the second operand for each element in the row.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each row operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::rows, "Vector length must match the number of rows in the tile");
    #pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[row]);
    }
}

/**
 * @brief Performs a column-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each column of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each column, using the vector element as 
 * the second operand for each element in the column.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each column operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::cols, "Vector length must match the number of columns in the tile");
    #pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[col]);
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

// const maps
/**
 * @brief Sets all elements of the destination tile to zero.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to one.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to positive infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to negative infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile, in base 2.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}
/**
 * @brief Applies the natural logarithm function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the natural logarithm function is applied.
 */
template<ducks::st::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}
/**
 * @brief Applies the logarithm base 2 function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the logarithm base 2 function is applied.
 */
template<ducks::st::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}
/**
 * @brief Applies the hyperbolic tangent function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the hyperbolic tangent function is applied.
 */
template<ducks::st::all T>
__device__ static inline void tanh(T &dst, const T &src) {
    unary_map<base_ops::tanh, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T tanh(const T &src) {
    T dst;
    tanh(dst, src);
    return dst;
}
/**
 * @brief Applies the absolute function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the absolute function is applied.
 */
template<ducks::st::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}
/**
 * @brief Applies the rectified linear unit function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the rectified linear unit function is applied.
 */
template<ducks::st::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}
/**
 * @brief Copies the elements of the source tile to the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source data to be copied.
 */
template<ducks::st::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps
/**
 * @brief Finds the maximum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Finds the minimum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Adds each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    add(dst, lhs, rhs);
    return dst;
}
template<ducks::st::all T, typename U>
__device__ static inline void operator+=(T &lhs, const U &rhs) {
    add(lhs, lhs, rhs);
}
/**
 * @brief Subtracts each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    sub(dst, lhs, rhs);
    return dst;
}
template<ducks::st::all T, typename U>
__device__ static inline void operator-=(T &lhs, const U &rhs) {
    sub(lhs, lhs, rhs);
}
/**
 * @brief Multiplies each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    mul(dst, lhs, rhs);
    return dst;
}
template<ducks::st::all T, typename U>
__device__ static inline void operator*=(T &lhs, const U &rhs) {
    mul(lhs, lhs, rhs);
}
/**
 * @brief Divides each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    div(dst, lhs, rhs);
    return dst;
}
template<ducks::st::all T, typename U>
__device__ static inline void operator/=(T &lhs, const U &rhs) {
    div(lhs, lhs, rhs);
}

// Row and col maps

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<ducks::st::all T, ducks::sv::all V>
__device__ static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

}