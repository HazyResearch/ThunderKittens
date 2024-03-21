#pragma once

#include <cuda_bf16.h>
#include <limits>
#include "base_types.cuh"

namespace kittens {

namespace base_ops {

/* ----------  CONST OPS  ---------- */

/**
 * @brief Struct representing the constant zero value.
 *
 * This struct provides a templated operation that returns the constant zero value
 * for various data types.
 */
struct zero {
    /**
     * @brief Returns the constant zero value for the specified type.
     *
     * @tparam T The data type for which to return the zero value.
     * @tparam args Additional arguments (unused).
     * @return constexpr T The constant zero value for the specified type.
     */
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::zero();      }
};

/**
 * @brief Struct representing the constant one value.
 *
 * This struct provides a templated operation that returns the constant one value
 * for various data types.
 */
struct one {
    /**
     * @brief Returns the constant one value for the specified type.
     *
     * @tparam T The data type for which to return the one value.
     * @tparam args Additional arguments (unused).
     * @return constexpr T The constant one value for the specified type.
     */
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
};

/**
 * @brief Struct representing the positive infinity value.
 *
 * This struct provides a templated operation that returns the positive infinity value
 * for various data types.
 */
struct pos_infty {
    /**
     * @brief Returns the positive infinity value for the specified type.
     *
     * @tparam T The data type for which to return the positive infinity value.
     * @tparam args Additional arguments (unused).
     * @return constexpr T The positive infinity value for the specified type.
     */
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};

/**
 * @brief Struct representing the negative infinity value.
 *
 * This struct provides a templated operation that returns the negative infinity value
 * for various data types.
 */
struct neg_infty {
    /**
     * @brief Returns the negative infinity value for the specified type.
     *
     * @tparam T The data type for which to return the negative infinity value.
     * @tparam args Additional arguments (unused).
     * @return constexpr T The negative infinity value for the specified type.
     */
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};


/* ----------  UNARY OPS  ---------- */

/**
 * @brief Struct providing exponential operations for various data types.
 *
 * This struct includes templated and specialized operations to perform
 * the exponential function on different data types using CUDA math functions.
 */
struct exp {
    /**
     * @brief General exponential operation for a given type.
     *
     * @tparam T The data type for the operation.
     * @param x The input value.
     * @return T The result of the exponential operation on the input value.
     */
    template<typename T> static __device__ inline T op(const T &x) { return exp(x); }

    /**
     * @brief Specialized exponential operation for float type using CUDA __expf function.
     *
     * @param x The input float value.
     * @return float The result of the exponential operation on the input value.
     */
    template<> __device__ inline float  op<float> (const float &x ) { return __expf(x); }

    /**
     * @brief Specialized exponential operation for float2 type using CUDA __expf function.
     *
     * @param x The input float2 value.
     * @return float2 The result of the exponential operation on the input value.
     */
    template<> __device__ inline float2 op<float2>(const float2 &x) { return float2{__expf(x.x), __expf(x.y)}; }

    /**
     * @brief Specialized exponential operation for bf16 type using custom hexp function.
     *
     * @param x The input bf16 value.
     * @return bf16 The result of the exponential operation on the input value.
     */
    template<> __device__ inline bf16   op<bf16>  (const bf16 &x  ) { return hexp(x); }

    /**
     * @brief Specialized exponential operation for bf16_2 type using custom h2exp function.
     *
     * @param x The input bf16_2 value.
     * @return bf16_2 The result of the exponential operation on the input value.
     */
    template<> __device__ inline bf16_2 op<bf16_2>(const bf16_2 &x) { return h2exp(x); }
};

/**
 * @brief Struct providing absolute value operations for various data types.
 *
 * This struct includes templated and specialized operations to perform
 * the absolute value function on different data types using CUDA math functions.
 */
struct abs {
    /**
     * @brief General absolute value operation for a given type.
     *
     * @tparam T The data type for the operation.
     * @param x The input value.
     * @return T The absolute value of the input.
     */
    template<typename T> static __device__ inline T op(const T &x) { return abs(x); }

    /**
     * @brief Specialized absolute value operation for float type using CUDA fabsf function.
     *
     * @param x The input float value.
     * @return float The absolute value of the input.
     */
    template<> __device__ inline float  op<float> (const float &x ) { return fabsf(x); }

    /**
     * @brief Specialized absolute value operation for float2 type.
     *
     * @param x The input float2 value.
     * @return float2 The absolute value of the input for each component.
     */
    template<> __device__ inline float2 op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }

    /**
     * @brief Specialized absolute value operation for bf16 type using CUDA __habs function.
     *
     * @param x The input bf16 value.
     * @return bf16 The absolute value of the input.
     */
    template<> __device__ inline bf16   op<bf16>  (const bf16 &x  ) { return __habs(x); }

    /**
     * @brief Specialized absolute value operation for bf16_2 type using CUDA __habs2 function.
     *
     * @param x The input bf16_2 value.
     * @return bf16_2 The absolute value of the input for each component.
     */
    template<> __device__ inline bf16_2 op<bf16_2>(const bf16_2 &x) { return __habs2(x); }
};

/**
 * @brief Struct implementing the ReLU (Rectified Linear Unit) operation.
 *
 * The ReLU operation is defined as the positive part of its argument. This struct
 * provides templated and specialized operations to perform the ReLU function on
 * different data types using CUDA math functions.
 */
struct relu {
    /**
     * @brief General ReLU operation for a given type.
     *
     * @tparam T The data type for the operation.
     * @param x The input value.
     * @return T The result of the ReLU operation on the input value.
     */
    template<typename T> static __device__ inline T op(const T &x) { return max(x, base_types::constants<T>::zero()); }

    /**
     * @brief Specialized ReLU operation for float type using CUDA max function.
     *
     * @param x The input float value.
     * @return float The result of the ReLU operation on the input value.
     */
    template<> __device__ inline float  op<float> (const float &x ) { return max(x, 0.f); }

    /**
     * @brief Specialized ReLU operation for float2 type using CUDA max function.
     *
     * @param x The input float2 value.
     * @return float2 The result of the ReLU operation on each component of the input value.
     */
    template<> __device__ inline float2 op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)}; }

    /**
     * @brief Specialized ReLU operation for bf16 type using CUDA __hmax function.
     *
     * @param x The input bf16 value.
     * @return bf16 The result of the ReLU operation on the input value.
     */
    template<> __device__ inline bf16   op<bf16>  (const bf16 &x  ) { return __hmax(x, base_types::constants<bf16>::zero()); }

    /**
     * @brief Specialized ReLU operation for bf16_2 type using CUDA __hmax2 function.
     *
     * @param x The input bf16_2 value.
     * @return bf16_2 The result of the ReLU operation on each component of the input value.
     */
    template<> __device__ inline bf16_2 op<bf16_2>(const bf16_2 &x) { return __hmax2(x, base_types::constants<bf16_2>::zero()); }
};

/**
 * @brief Struct providing a copy operation.
 *
 * This struct includes a templated operation to return the input value as is,
 * which can be used in various computational scenarios where a direct copy is needed.
 */
struct copy {
    /**
     * @brief Returns the input value unchanged.
     *
     * @tparam T The data type of the input.
     * @param a The input value.
     * @return T The same value as the input.
     */
    template<typename T> static __device__ inline T op(const T &a) { return a; }
};

/**
 * @brief Struct providing an operation to return the second of two input values.
 *
 * This struct includes a templated operation that returns the second input value,
 * which can be used in scenarios where the second value is the desired output after
 * some conditional check or operation.
 */
struct copy2 {
    /**
     * @brief Returns the second input value.
     *
     * @tparam T The data type of the inputs.
     * @param a The first input value.
     * @param b The second input value.
     * @return T The second input value.
     */
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return b; }
};

/**
 * @brief Struct providing a binary addition operation.
 *
 * This struct includes a templated operation that performs the addition of two input values,
 * which can be used in various mathematical and computational contexts.
 */
struct sum {
    /**
     * @brief Performs addition on two input values.
     *
     * @tparam T The data type of the inputs.
     * @param a The first input value.
     * @param b The second input value.
     * @return T The sum of the input values.
     */
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a+b; }
};

/**
 * @brief Struct providing a binary subtraction operation.
 *
 * This struct includes a templated operation that performs the subtraction of two input values,
 * which can be used in various mathematical and computational contexts.
 */
struct sub {
    /**
     * @brief Performs subtraction on two input values.
     *
     * @tparam T The data type of the inputs.
     * @param a The first input value.
     * @param b The second input value.
     * @return T The result of subtracting the second input value from the first.
     */
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a-b; }
};

/**
 * @brief Struct providing a binary multiplication operation.
 *
 * This struct includes a templated operation that performs the multiplication of two input values,
 * which can be used in various mathematical and computational contexts.
 */
struct mul {
    /**
     * @brief Performs multiplication on two input values.
     *
     * @tparam T The data type of the inputs.
     * @param a The first input value.
     * @param b The second input value.
     * @return T The product of the input values.
     */
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a*b; }
};

/**
 * @brief Struct providing a binary division operation.
 *
 * This struct includes a templated operation that performs the division of two input values,
 * which can be used in various mathematical and computational contexts.
 */
struct div {
    /**
     * @brief Performs division on two input values.
     *
     * @tparam T The data type of the inputs.
     * @param a The dividend input value.
     * @param b The divisor input value.
     * @return T The result of dividing the first input value by the second.
     */
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a/b; }
};

/* ----------  BINARY OPS  ---------- */
// ... (rest of the code remains unchanged)
