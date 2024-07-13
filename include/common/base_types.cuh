/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types.
 * 
 * This file is a bunch of utilities for going back and forth between different types.
 * 
 * Many of them are for the compiler, so as to clean up the code. It unfortunately
 * seems necessary when we have types we really care about that are less than word width.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <string>
#include <bit>

namespace kittens {

/**
 * @brief Bfloat16 floating-point type.
 */
using bf16 = __nv_bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = __half;
/**
 * @brief Packed word of two bfloat16 floating-point values.
 */
using bf16_2 = __nv_bfloat162;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = __half2;

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2>; // could add half_2 later if implemented.
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half>; // could add half_2 later if implemented.

} // namespace base_types
} // namespace ducks

/**
 * @namespace base_types
 *
 * @brief A namespace for ThunderKittens basic data types.
 */
namespace base_types {

/**
 * @brief Provides compile-time constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    /**
     * @brief Zero
     * @return Constexpr zero with type T
     */
    static __device__ inline constexpr T zero()      { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static __device__ inline constexpr T one()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 one()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<__nv_bfloat16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __device__ inline constexpr bf16 one()       { return std::bit_cast<__nv_bfloat16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __device__ inline constexpr bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline constexpr bf16_2 one()       { return bf16_2{constants<bf16>::one(),       constants<bf16>::one()};       }
    static __device__ inline constexpr bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline constexpr bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __device__ inline constexpr half zero()      { return std::bit_cast<__half>(uint16_t(0x0000)); }
    static __device__ inline constexpr half one()       { return std::bit_cast<__half>(uint16_t(0x3C00)); }
    static __device__ inline constexpr half pos_infty() { return std::bit_cast<__half>(uint16_t(0x7C00)); }
    static __device__ inline constexpr half neg_infty() { return std::bit_cast<__half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __device__ inline constexpr half_2 zero()      { return half_2{constants<half>::zero(),      constants<half>::zero()};      }
    static __device__ inline constexpr half_2 one()       { return half_2{constants<half>::one(),       constants<half>::one()};       }
    static __device__ inline constexpr half_2 pos_infty() { return half_2{constants<half>::pos_infty(), constants<half>::pos_infty()}; }
    static __device__ inline constexpr half_2 neg_infty() { return half_2{constants<half>::neg_infty(), constants<half>::neg_infty()}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __device__ inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __device__ inline constexpr T pack(const bf16 &i);
};
template<> struct packing<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<half> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<float> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int2> {
    static __device__ inline constexpr int num() { return 2; }
};
template<> struct packing<float4> {
    static __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __device__ inline constexpr int num() { return 4; }
};

/**
 * @brief Provides templated functionality to convert between different types.
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
template<> struct convertor<float, bf16> {
    static __device__ inline float convert(const bf16 & u) {
        return 	__bfloat162float(u);
    }
};
template<> struct convertor<bf16, float> {
    static __device__ inline bf16 convert(const float & u) {
        return 	__float2bfloat16_rn(u);
    }
};
template<> struct convertor<float2, bf16_2> {
    static __device__ inline float2 convert(const bf16_2 & u) {
        return 	__bfloat1622float2(u);
    }
};
template<> struct convertor<bf16_2, float2> {
    static __device__ inline bf16_2 convert(const float2 & u) {
        return 	__float22bfloat162_rn(u);
    }
};
template<> struct convertor<float, half> {
    static __device__ inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __device__ inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __device__ inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __device__ inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __device__ inline bf16 convert(const half & u) {
        return __float2bfloat16_rn(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __device__ inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __device__ inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __device__ inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};


}
}
