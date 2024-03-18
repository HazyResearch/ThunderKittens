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

using bf16 = __nv_bfloat16;
using half = __half;
using bf16_2 = __nv_bfloat162;
using half_2 = __half2;

namespace ducks {
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2>; // could add half_2 later if implemented.

} // namespace base_types
} // namespace ducks

namespace base_types {

template<typename T> struct constants {
    static __device__ inline constexpr T zero()      { return T{0}; }
    static __device__ inline constexpr T one()       { return T{1}; }
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
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

// how many are packed?
template<typename T> struct packing {
    static __device__ inline constexpr int num() { return 1; }
};
template<> struct packing<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<half> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<float> {
    static __device__ inline constexpr int num() { return 1; }
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
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


template<typename T, typename U> struct convertor {
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

}
}
