/**
 * @file
 * @brief Warp-scope maps on shared vectors.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../common/common.dp.hpp"
#include "../../../../types/types.dp.hpp"
#include <cmath>

namespace kittens {

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector in which to store the result.
 * @param src[in] Source vector to apply the unary operation.
 */
template<typename op, ducks::sv::all T>
static inline void unary_op(T &dst, const T &src) {
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
    for(int cur = kittens::laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur]);
    }
}
/**
 * @brief Perform a binary operation on two shared vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, ducks::sv::all T>
static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(lhs[cur], rhs[cur]);
    }
}
/**
 * @brief Perform a binary operation on a shared vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, ducks::sv::all T>
static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<ducks::sv::all T>
static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<ducks::sv::all T>
static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<ducks::sv::all T>
static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<ducks::sv::all T>
static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one shared vector to another.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<ducks::sv::all T, typename U>
static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
template<ducks::sv::all T>
static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}
/**
 * @brief Applies the exponential function element-wise to a shared vector, in base 2.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
template<ducks::sv::all T>
static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}
/**
 * @brief Applies the natural logarithm function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm values will be stored.
 * @param src[in] Source vector to apply the logarithm function to.
 */
template<ducks::sv::all T>
static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
template<ducks::sv::all T>
static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm base 2 values will be stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template<ducks::sv::all T>
static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
template<ducks::sv::all T>
static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}
/**
 * @brief Applies the absolute value function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<ducks::sv::all T>
static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<ducks::sv::all T>
static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<ducks::sv::all T>
static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}
template<ducks::sv::all T>
static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<ducks::sv::all T, typename U>
static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Computes the element-wise minimum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<ducks::sv::all T, typename U>
static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Computes the element-wise sum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<ducks::sv::all T, typename U>
static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    add(dst, lhs, rhs);
    return dst;
}
template<ducks::sv::all T, typename U>
static inline void operator+=(T &lhs, const U &rhs) {
    add(lhs, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<ducks::sv::all T, typename U>
static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    sub(dst, lhs, rhs);
    return dst;
}
template<ducks::sv::all T, typename U>
static inline void operator-=(T &lhs, const U &rhs) {
    sub(lhs, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<ducks::sv::all T, typename U>
static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    mul(dst, lhs, rhs);
    return dst;
}
template<ducks::sv::all T, typename U>
static inline void operator*=(T &lhs, const U &rhs) {
    mul(lhs, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<ducks::sv::all T, typename U>
static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    div(dst, lhs, rhs);
    return dst;
}
template<ducks::sv::all T, typename U>
static inline void operator/=(T &lhs, const U &rhs) {
    div(lhs, lhs, rhs);
}

}
