#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/**
 * @brief Perform a unary operation on a vector.
 *
 * @tparam op The unary operation to perform.
 * @tparam T The type of the vector.
 * @param dst The destination vector where the result is stored.
 * @param src The source vector to perform the operation on.
 */
template<typename op, typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void unary_op(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j]);
        }
    }
}

/**
 * @brief Perform a binary operation on two vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst The destination vector where the result is stored.
 * @param lhs The left-hand side vector for the operation.
 * @param rhs The right-hand side vector for the operation.
 */
template<typename op, typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(lhs[i][j], rhs[i][j]);
        }
    }
}

/**
 * @brief Perform a binary operation on a vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst The destination vector where the result is stored.
 * @param src The source vector for the operation.
 * @param param The scalar parameter for the operation.
 */
template<typename op, typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j], param);
        }
    }
}

/**
 * @brief Perform a binary operation on a vector and an unpacked scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst The destination vector where the result is stored.
 * @param src The source vector for the operation.
 * @param param The unpacked scalar parameter for the operation.
 */
template<typename op, typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<typename T, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
template<typename T, typename U, typename std::enable_if<is_rt_vec_type<T>::value, int>::type = 0>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}

}
