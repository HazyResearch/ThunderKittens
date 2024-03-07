#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
    
/* ----------  Vector Operations  ---------- */

/**
 * @brief Perform a unary operation on a vector.
 *
 * @tparam op The unary operation to perform.
 * @tparam T The type of the vector.
 * @param dst The destination vector where the result is stored.
 * @param src The source vector to perform the operation on.
 */
template<typename op, rt_vec_type T>
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
template<typename op, rt_vec_type T>
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
template<typename op, rt_vec_type T>
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
template<typename op, rt_vec_type T>
__device__ static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

// col vectors

/**
 * @brief Sets all elements of a register vector to zero.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector to be set to zero.
 */
template<rt_vec_type T>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to one.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector to be set to one.
 */
template<rt_vec_type T>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to positive infinity.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector to be set to positive infinity.
 */
template<rt_vec_type T>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to negative infinity.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector to be set to negative infinity.
 */
template<rt_vec_type T>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Copies the elements from one register vector to another.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the source vector.
 * @param dst Destination vector where the elements will be copied to.
 * @param src Source vector to copy the elements from.
 */
template<rt_vec_type T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}

/**
 * @brief Applies the exponential function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector where the exponential values will be stored.
 * @param src Source vector to apply the exponential function to.
 */
template<rt_vec_type T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}

/**
 * @brief Applies the absolute value function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector where the absolute values will be stored.
 * @param src Source vector to apply the absolute value function to.
 */
template<rt_vec_type T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}

/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst Destination vector where the ReLU values will be stored.
 * @param src Source vector to apply the ReLU function to.
 */
template<rt_vec_type T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

/**
 * @brief Computes the element-wise maximum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the maximum values will be stored.
 * @param lhs First vector for the maximum operation.
 * @param rhs Second vector for the maximum operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}

/**
 * @brief Computes the element-wise minimum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the minimum values will be stored.
 * @param lhs First vector for the minimum operation.
 * @param rhs Second vector for the minimum operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}

/**
 * @brief Computes the element-wise sum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the sum values will be stored.
 * @param lhs First vector for the sum operation.
 * @param rhs Second vector for the sum operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}

/**
 * @brief Computes the element-wise difference of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the difference values will be stored.
 * @param lhs First vector for the difference operation.
 * @param rhs Second vector for the difference operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}

/**
 * @brief Computes the element-wise product of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the product values will be stored.
 * @param lhs First vector for the product operation.
 * @param rhs Second vector for the product operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}

/**
 * @brief Computes the element-wise division of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst Destination vector where the division values will be stored.
 * @param lhs First vector for the division operation.
 * @param rhs Second vector for the division operation.
 */
template<rt_vec_type T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}

}