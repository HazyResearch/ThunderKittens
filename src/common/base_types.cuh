#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <string>
#include <bit>

/*

This file is a bunch of utilities for going back and forth between different types

Many of them are for the compiler, so as to clean up the code. It unfortunately
seems necessary when we have types we really care about that are less than word width.

*/

namespace kittens {

/**
 * @brief Half-precision floating-point type.
 */
using half = __half;

/**
 * @brief Bfloat16 floating-point type.
 */
using bf16 = __nv_bfloat16;

/**
 * @brief Vector of two half-precision floating-point values.
 */
using half_2 = __half2;

/**
 * @brief Vector of two bfloat16 floating-point values.
 */
using bf16_2 = __nv_bfloat162;

/**
 * @brief Concept to check if a type is a packed vector type.
 */
template<typename T>
concept packed_type = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2>; // could add half_2 later if implemented.

namespace base_types {

/**
 * @brief Provides constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    static __device__ inline constexpr T zero()      { return T{0}; }
    static __device__ inline constexpr T one()       { return T{1}; }
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};

/**
 * @brief Specialization of constants for float2 type.
 */
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 one()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};

/**
 * @brief Specialization of constants for bf16 type.
 */
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<__nv_bfloat16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __device__ inline constexpr bf16 one()       { return std::bit_cast<__nv_bfloat16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0xFF80)); }
};

/**
 * @brief Specialization of constants for bf16_2 type.
 */
template<> struct constants<bf16_2> {
    static __device__ inline constexpr bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline constexpr bf16_2 one()       { return bf16_2{constants<bf16>::one(),       constants<bf16>::one()};       }
    static __device__ inline constexpr bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline constexpr bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief Returns the number of elements packed together.
     *
     * @return constexpr int Number of elements.
     */
    static __device__ inline constexpr int num() { return 1; }
};

/**
 * @brief Specialization of packing for bf16 type.
 */
template<> struct packing<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = bf16_2;
    /**
     * @brief Packs a single bf16 element into a bf16_2 vector.
     *
     * @param i[in] The bf16 element to pack.
     * @return constexpr bf16_2 The packed bf16_2 vector.
     */
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};

/**
 * @brief Specialization of packing for half type.
 */
template<> struct packing<half> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = half_2;
    /**
     * @brief Packs a single half element into a half_2 vector.
     *
     * @param i[in] The half element to pack.
     * @return constexpr half_2 The packed half_2 vector.
     */
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};

/**
 * @brief Specialization of packing for float type.
 */
template<> struct packing<float> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = float2;
    /**
     * @brief Packs a single float element into a float2 vector.
     *
     * @param i[in] The float element to pack.
     * @return constexpr float2 The packed float2 vector.
     */
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};

/**
 * @brief Specialization of packing for bf16_2 type.
 */
template<> struct packing<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    /**
     * @brief Packs a single bf16 element into a bf16_2 vector by replicating it.
     *
     * @param i[in] The bf16 element to pack.
     * @return constexpr bf16_2 The packed bf16_2 vector.
     */
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};

/**
 * @brief Specialization of packing for half_2 type.
 */
template<> struct packing<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    /**
     * @brief Packs a single half element into a half_2 vector by replicating it.
     *
     * @param i[in] The half element to pack.
     * @return constexpr half_2 The packed half_2 vector.
     */
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};

/**
 * @brief Specialization of packing for float2 type.
 */
template<> struct packing<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    /**
     * @brief Packs a single float element into a float2 vector by replicating it.
     *
     * @param i[in] The float element to pack.
     * @return constexpr float2 The packed float2 vector.
     */
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};

/**
 * @brief Specialization of packing for int2 type.
 */
template<> struct packing<int2> {
    static __device__ inline constexpr int num() { return 2; }
};

/**
 * @brief Specialization of packing for float4 type.
 */
template<> struct packing<float4> {
    static __device__ inline constexpr int num() { return 4; }
};

/**
 * @brief Specialization of packing for int4 type.
 */
template<> struct packing<int4> {
    static __device__ inline constexpr int num() { return 4; }
};

/**
 * @brief Provides functionality to convert between different types.
 *
 * @tparam T The target type for conversion.
 * @tparam U The source type for conversion.
 */
template<typename T, typename U> struct convertor {
    /**
     * @brief Converts a value of type U to type T.
     *
     * @param u[in] The value of type U to convert.
     * @return T The converted value of type T.
     */
    static __device__ inline T convert(const U & u) {
        return (T)u;
    }
};

/**
 * @brief Specialization of convertor for converting bf16 to float.
 */
template<> struct convertor<float, bf16> {
    static __device__ inline float convert(const bf16 & u) {
        return 	__bfloat162float(u);
    }
};

/**
 * @brief Specialization of convertor for converting float to bf16.
 */
template<> struct convertor<bf16, float> {
    static __device__ inline bf16 convert(const float & u) {
        return 	__float2bfloat16_rn(u);
    }
};

/**
 * @brief Specialization of convertor for converting bf16_2 to float2.
 */
template<> struct convertor<float2, bf16_2> {
    static __device__ inline float2 convert(const bf16_2 & u) {
        return 	__bfloat1622float2(u);
    }
};

/**
 * @brief Specialization of convertor for converting float2 to bf16_2.
 */
template<> struct convertor<bf16_2, float2> {
    static __device__ inline bf16_2 convert(const float2 & u) {
        return 	__float22bfloat162_rn(u);
    }
};

}
}
