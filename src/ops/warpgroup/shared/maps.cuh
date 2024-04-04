#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace warpgroup {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Applies a unary operation to each element of a shared tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
template<typename op, ducks::st::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    int lane = threadIdx.x % (WARP_THREADS * 4); 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += (WARP_THREADS * 4)) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)]);
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {

    int lane = threadIdx.x % (WARP_THREADS * 4); 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += (WARP_THREADS * 4)) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols);  
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)], param);
    }
}
/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {

    int lane = threadIdx.x % (WARP_THREADS * 4);
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += (WARP_THREADS * 4)) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(lhs.data[col + (row * dst.cols)], rhs.data[col + (row * dst.cols)]);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::st::all T>
__device__ static inline void zero(T &dst) {
    warpgroup::unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::st::all T>
__device__ static inline void one(T &dst) {
    warpgroup::unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::st::all T>
__device__ static inline void pos_infty(T &dst) {
    warpgroup::unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::st::all T>
__device__ static inline void neg_infty(T &dst) {
    warpgroup::unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<ducks::st::all T>
__device__ static inline void exp(T &dst, const T &src) {
    warpgroup::unary_map<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the absolute value function on.
 */
template<ducks::st::all T>
__device__ static inline void abs(T &dst, const T &src) {
    warpgroup::unary_map<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the ReLU function on.
 */
template<ducks::st::all T>
__device__ static inline void relu(T &dst, const T &src) {
    warpgroup::unary_map<base_ops::relu, T>(dst, src);
}
/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to copy from.
 */
template<ducks::st::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    warpgroup::bin_map<base_ops::copy2, T>(dst, src);
}

/**
 * @brief Applies the max operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::st::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Applies the min operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::st::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile or scalar for the addition.
 */
template<ducks::st::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile or scalar for the subtraction.
 */
template<ducks::st::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
template<ducks::st::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template<ducks::st::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    warpgroup::bin_map<base_ops::div, T>(dst, lhs, rhs);
}

}
}