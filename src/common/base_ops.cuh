#pragma once

#include <cuda_bf16.h>
#include <limits>
#include "base_types.cuh"

namespace kittens {

namespace base_ops {

/* ----------  CONST OPS  ---------- */

/**
 * @brief Represents the zero constant operation.
 *
 * This operation returns the zero value of the specified type.
 *
 * @tparam T The data type for which to return the zero value.
 * @return The zero value of type T.
 */
struct zero {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::zero();      }
};

/**
 * @brief Represents the one constant operation.
 *
 * This operation returns the one value of the specified type.
 *
 * @tparam T The data type for which to return the one value.
 * @return The one value of type T.
 */
struct one {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
};

/**
 * @brief Represents the positive infinity constant operation.
 *
 * This operation returns the positive infinity value of the specified type.
 *
 * @tparam T The data type for which to return the positive infinity value.
 * @return The positive infinity value of type T.
 */
struct pos_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};

/**
 * @brief Represents the negative infinity constant operation.
 *
 * This operation returns the negative infinity value of the specified type.
 *
 * @tparam T The data type for which to return the negative infinity value.
 * @return The negative infinity value of type T.
 */
struct neg_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};


/* ----------  UNARY OPS  ---------- */

/**
 * @brief Exponential function operation.
 *
 * This operation calculates the exponential of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x The input value.
 * @return The exponential of the input value.
 */
struct exp {
    template<typename T> static __device__ inline T op(const T &x) { return exp(x); }
};
template<> __device__ inline float  exp::op<float> (const float &x ) { return __expf(x);                        }
template<> __device__ inline float2 exp::op<float2>(const float2 &x) { return float2{__expf(x.x), __expf(x.y)}; }
template<> __device__ inline bf16   exp::op<bf16>  (const bf16 &x  ) { return __float2bfloat16(__expf(__bfloat162float(x))); }
template<> __device__ inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) {
    bf16 low = __float2bfloat16(__expf(__bfloat162float(x.x)));
    bf16 high = __float2bfloat16(__expf(__bfloat162float(x.y)));
    return bf16_2{low, high};
}

/**
 * @brief Absolute value operation.
 *
 * This operation calculates the absolute value of the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x The input value.
 * @return The absolute value of the input.
 */
struct abs {
    template<typename T> static __device__ inline T op(const T &x) { return abs(x); }
};
template<> __device__ inline float  abs::op<float> (const float &x ) { return fabsf(x);                       }
template<> __device__ inline float2 abs::op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }
template<> __device__ inline bf16   abs::op<bf16>  (const bf16 &x  ) { return __float2bfloat16(fabsf(__bfloat162float(x))); }
template<> __device__ inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) {
    bf16 low = __float2bfloat16(fabsf(__bfloat162float(x.x)));
    bf16 high = __float2bfloat16(fabsf(__bfloat162float(x.y)));
    return bf16_2{low, high};
}

/**
 * @brief Rectified Linear Unit (ReLU) operation.
 *
 * This operation applies the ReLU function to the input, which is the
 * maximum of zero and the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x The input value.
 * @return The result of ReLU function applied to the input.
 */
struct relu {
    template<typename T> static __device__ inline T op(const T &x) { return max(x, base_types::constants<T>::zero()); }
};
template<> __device__ inline float  relu::op<float> (const float &x ) { return max(x, 0.f);                                  }
template<> __device__ inline float2 relu::op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)};         }
template<> __device__ inline bf16 relu::op<bf16>(const bf16 &x) {
    return max(x, __float2bfloat16(static_cast<float>(0)));
}
template<> __device__ inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) {
    float2 f2 = make_float2(
        fmaxf(__bfloat162float(x.x), 0.0f),
        fmaxf(__bfloat162float(x.y), 0.0f)
    );
    return bf16_2{__float2bfloat16(f2.x), __float2bfloat16(f2.y)};
}

/**
 * @brief Copy operation.
 *
 * This operation returns the input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a The input value.
 * @return The same value as the input.
 */
struct copy { // for non-compile-time setters.
    template<typename T> static __device__ inline T op(const T &a) { return a; }
};


/* ----------  BINARY OPS  ---------- */

/**
 * @brief Copy2 operation.
 *
 * This operation returns the second input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value (ignored).
 * @param b The second input value.
 * @return The same value as the second input.
 */
struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return b; }
};

/**
 * @brief Sum operation.
 *
 * This operation calculates the sum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The sum of the input values.
 */
struct sum {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a+b; }
};
template<> __device__ inline float2 sum::op<float2>(const float2 &a, const float2 &b) { return float2{a.x+b.x, a.y+b.y}; }
template<> __device__ inline bf16 sum::op<bf16>(const bf16 &a, const bf16 &b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}
template<> __device__ inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low = __bfloat162float(a.x);
    float b_low = __bfloat162float(b.x);
    float a_high = __bfloat162float(a.y);
    float b_high = __bfloat162float(b.y);
    return bf16_2{__float2bfloat16(a_low + b_low), __float2bfloat16(a_high + b_high)};
}

/**
 * @brief Subtraction operation.
 *
 * This operation calculates the difference between two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The difference between the input values.
 */
struct sub {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a-b; }
};
template<> __device__ inline float2 sub::op<float2>(const float2 &a, const float2 &b) { return float2{a.x-b.x, a.y-b.y}; }
template<> __device__ inline bf16 sub::op<bf16>(const bf16 &a, const bf16 &b) {
    return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}
template<> __device__ inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low = __bfloat162float(a.x);
    float b_low = __bfloat162float(b.x);
    float a_high = __bfloat162float(a.y);
    float b_high = __bfloat162float(b.y);
    return bf16_2{__float2bfloat16(a_low - b_low), __float2bfloat16(a_high - b_high)};
}

/**
 * @brief Multiplication operation.
 *
 * This operation calculates the product of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The product of the input values.
 */
struct mul {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a*b; }
};
template<> __device__ inline float2 mul::op<float2>(const float2 &a, const float2 &b) { return float2{a.x*b.x, a.y*b.y}; }
template<> __device__ inline bf16 mul::op<bf16>(const bf16 &a, const bf16 &b) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}
template<> __device__ inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low = __bfloat162float(a.x);
    float b_low = __bfloat162float(b.x);
    float a_high = __bfloat162float(a.y);
    float b_high = __bfloat162float(b.y);
    return bf16_2{__float2bfloat16(a_low * b_low), __float2bfloat16(a_high * b_high)};
}

/**
 * @brief Division operation.
 *
 * This operation calculates the quotient of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The quotient of the input values.
 */
struct div {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a/b; }
};
template<> __device__ inline float2 div::op<float2>(const float2 &a, const float2 &b) { return float2{a.x/b.x, a.y/b.y}; }
template<> __device__ inline bf16 div::op<bf16>(const bf16 &a, const bf16 &b) {
    float a_float = __bfloat162float(a);
    float b_float = __bfloat162float(b);
    return __float2bfloat16(a_float / b_float);
}
template<> __device__ inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low_float = __bfloat162float(a.x);
    float b_low_float = __bfloat162float(b.x);
    float a_high_float = __bfloat162float(a.y);
    float b_high_float = __bfloat162float(b.y);
    bf16 low = __float2bfloat16(a_low_float / b_low_float);
    bf16 high = __float2bfloat16(a_high_float / b_high_float);
    return bf16_2{low, high};
}

/**
 * @brief Maximum operation.
 *
 * This operation calculates the maximum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The maximum of the input values.
 */
struct max {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::max(a, b); }
};
template<>  __device__ inline float2 max::op<float2>(const float2 &a, const float2 &b) { return float2{::max(a.x, b.x), ::max(a.y, b.y)}; }
template<>  __device__ inline bf16   max::op<bf16>  (const bf16   &a, const bf16   &b) {
    float a_float = __bfloat162float(a);
    float b_float = __bfloat162float(b);
    return __float2bfloat16(fmaxf(a_float, b_float));
}
template<>  __device__ inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low_float = __bfloat162float(a.x);
    float b_low_float = __bfloat162float(b.x);
    float a_high_float = __bfloat162float(a.y);
    float b_high_float = __bfloat162float(b.y);
    bf16 low = __float2bfloat16(fmaxf(a_low_float, b_low_float));
    bf16 high = __float2bfloat16(fmaxf(a_high_float, b_high_float));
    return bf16_2{low, high};
}

/**
 * @brief Minimum operation.
 *
 * This operation calculates the minimum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @return The minimum of the input values.
 */
struct min {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::min(a, b); }
};
template<> __device__ inline bf16 min::op<bf16>(const bf16 &a, const bf16 &b) {
    float a_float = __bfloat162float(a);
    float b_float = __bfloat162float(b);
    return __float2bfloat16(fminf(a_float, b_float));
}
template<> __device__ inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    float a_low_float = __bfloat162float(a.x);
    float b_low_float = __bfloat162float(b.x);
    float a_high_float = __bfloat162float(a.y);
    float b_high_float = __bfloat162float(b.y);
    bf16 low = __float2bfloat16(fminf(a_low_float, b_low_float));
    bf16 high = __float2bfloat16(fminf(a_high_float, b_high_float));
    return bf16_2{low, high};
}


/* ----------  TERNARY OPS  ---------- */

/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The second input value.
 * @param c The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxBtC {
    template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, b), c);
    }
};

/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
 * This is particularly useful for attention mechanisms in neural networks.
 *
 * @tparam T The data type of the input and output values.
 * @param a The first input value.
 * @param b The third input value to be added.
 * @param c The second input value.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxCtB { // this is the one needed for attention
    template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, c), b);
    }
};

}

}
