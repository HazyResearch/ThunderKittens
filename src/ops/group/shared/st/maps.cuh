/**
 * @file
 * @brief Group maps on shared tiles.
 */

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

    int lane = laneid(); 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += GROUP_THREADS) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)]);
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

    int lane = laneid(); 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += GROUP_THREADS) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols);  
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)], param);
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

    int lane = laneid(); 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += GROUP_THREADS) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(lhs.data[col + (row * dst.cols)], rhs.data[col + (row * dst.cols)]);
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

}