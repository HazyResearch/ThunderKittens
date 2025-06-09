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

#ifdef KITTENS_HOPPER
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#endif

#include <string>
#include <bit>
#include <sycl/ext/intel/math.hpp>

namespace kittens {

/**
 * @brief Bfloat16 floating-point type.
 */
using bf16 = sycl::ext::oneapi::bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = sycl::half;
/**
 * @brief Packed word of two bfloat16 floating-point values.
 */
using bf16_2 = sycl::vec<sycl::ext::oneapi::bfloat16, 2>;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = sycl::half2;
#ifdef KITTENS_HOPPER
/**
 * @brief float8 floating-point type.
 */
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
/**
 * @brief 2-packed float8 floating-point type.
 */
using fp8e4m3_2 = __nv_fp8x2_e4m3;
using fp8e5m2_2 = __nv_fp8x2_e5m2;
/**
 * @brief 4-packed float8 floating-point type.
 */
using fp8e4m3_4 = __nv_fp8x4_e4m3;
using fp8e5m2_4 = __nv_fp8x4_e5m2;
#endif

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

#ifdef KITTENS_HOPPER
template <typename T>
concept T2 =
    std::is_same_v<T, sycl::float2> || std::is_same_v<T, bf16_2> ||
    std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4> ||
    std::is_same_v<T, fp8e5m2_4>; // could add half_2 later if implemented.
template <typename T>
concept T1 =
    std::is_same_v<T, float> || std::is_same_v<T, bf16> ||
    std::is_same_v<T, sycl::half> || std::is_same_v<T, fp8e4m3> ||
    std::is_same_v<T, fp8e5m2>; // could add half_2 later if implemented.
#else
template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half>;
#endif

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
    static inline constexpr T zero()      { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static inline constexpr T one()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template <> struct constants<sycl::float2> {
    static inline constexpr sycl::float2 zero() {
        return sycl::float2{0.f, 0.f};
    }
    static inline constexpr sycl::float2 one() {
        return sycl::float2{1.f, 1.f};
    }
    static inline constexpr sycl::float2 pos_infty() {
        return sycl::float2{constants<float>::pos_infty(),
                            constants<float>::pos_infty()};
    }
    static inline constexpr sycl::float2 neg_infty() {
        return sycl::float2{constants<float>::neg_infty(),
                            constants<float>::neg_infty()};
    }
};
template<> struct constants<bf16> {
    static inline constexpr bf16 zero() {
        return std::bit_cast<sycl::ext::oneapi::bfloat16>(uint16_t(0x0000));
    } // unfortunately __float2bf16_rn is not constexpr
    static inline constexpr bf16 one() {
        return std::bit_cast<sycl::ext::oneapi::bfloat16>(uint16_t(0x3F80));
    }
    static inline constexpr bf16 pos_infty() {
        return std::bit_cast<sycl::ext::oneapi::bfloat16>(uint16_t(0x7F80));
    }
    static inline constexpr bf16 neg_infty() {
        return std::bit_cast<sycl::ext::oneapi::bfloat16>(uint16_t(0xFF80));
    }
};
template<> struct constants<bf16_2> {
    static inline constexpr bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static inline constexpr bf16_2 one()       { return bf16_2{constants<bf16>::one(),       constants<bf16>::one()};       }
    static inline constexpr bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static inline constexpr bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template <> struct constants<sycl::half> {
    static inline constexpr sycl::half zero() {
        return std::bit_cast<sycl::half>(uint16_t(0x0000));
    }
    static inline constexpr sycl::half one() {
        return std::bit_cast<sycl::half>(uint16_t(0x3C00));
    }
    static inline constexpr sycl::half pos_infty() {
        return std::bit_cast<sycl::half>(uint16_t(0x7C00));
    }
    static inline constexpr sycl::half neg_infty() {
        return std::bit_cast<sycl::half>(uint16_t(0xFC00));
    }
};
template<> struct constants<half_2> {
    static inline constexpr half_2 zero() {
        return half_2{constants<sycl::half>::zero(),
                      constants<sycl::half>::zero()};
    }
    static inline constexpr half_2 one() {
        return half_2{constants<sycl::half>::one(),
                      constants<sycl::half>::one()};
    }
    static inline constexpr half_2 pos_infty() {
        return half_2{constants<sycl::half>::pos_infty(),
                      constants<sycl::half>::pos_infty()};
    }
    static inline constexpr half_2 neg_infty() {
        return half_2{constants<sycl::half>::neg_infty(),
                      constants<sycl::half>::neg_infty()};
    }
};
#ifdef KITTENS_HOPPER
template<> struct constants<fp8e4m3> {
    static inline constexpr fp8e4m3 zero() { return std::bit_cast<__nv_fp8_e4m3>(uint8_t(0x00)); }
    static inline constexpr fp8e4m3 one() { return std::bit_cast<__nv_fp8_e4m3>(uint8_t(0x38)); }
};
template<> struct constants<fp8e4m3_2> {
    static inline constexpr fp8e4m3_2 zero() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x0000)); }
    static inline constexpr fp8e4m3_2 one() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x3838)); }
};
template<> struct constants<fp8e4m3_4> {
    static inline constexpr fp8e4m3_4 zero() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x00000000)); }
    static inline constexpr fp8e4m3_4 one() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x38383838)); }
};
template<> struct constants<fp8e5m2> {
    static inline constexpr fp8e5m2 zero() { return std::bit_cast<__nv_fp8_e5m2>(uint8_t(0x00)); }
    static inline constexpr fp8e5m2 one() { return std::bit_cast<__nv_fp8_e5m2>(uint8_t(0x3C)); }
};
template<> struct constants<fp8e5m2_2> {
    static inline constexpr fp8e5m2_2 zero() { return std::bit_cast<fp8e5m2_2>(uint16_t(0x0000)); }
    static inline constexpr fp8e5m2_2 one() { return std::bit_cast<fp8e5m2_2>(uint16_t(0x3C3C)); }
};
template<> struct constants<fp8e5m2_4> {
    static inline constexpr fp8e5m2_4 zero() { return std::bit_cast<fp8e5m2_4>(uint32_t(0x00000000)); }
    static inline constexpr fp8e5m2_4 one() { return std::bit_cast<fp8e5m2_4>(uint32_t(0x3C3C3C3C)); }
};
#endif

template<> struct constants<int> {
    static inline constexpr int zero()      { return 0; }
    static inline constexpr int one()       { return 1; }
};
template <> struct constants<sycl::int2> {
    static inline constexpr sycl::int2 zero() { return sycl::int2{0, 0}; }
    static inline constexpr sycl::int2 one() { return sycl::int2{1, 1}; }
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
    static inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static inline constexpr T pack(const bf16 &i);
};
template<> struct packing<bf16> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template <> struct packing<sycl::half> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = sycl::half;
    using packed_type = half_2;
    static inline constexpr half_2 pack(const sycl::half &i) {
        return half_2{i, i};
    }
};
template<> struct packing<half_2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = sycl::half;
    using packed_type = half_2;
    static inline constexpr half_2 pack(const sycl::half &i) {
        return half_2{i, i};
    } // this replication makes code cleaner later.
};
template<> struct packing<float> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = sycl::float2;
    static inline constexpr sycl::float2 pack(const float &i) {
        return sycl::float2{i, i};
    }
};
template <> struct packing<sycl::float2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = sycl::float2;
    static inline constexpr sycl::float2 pack(const float &i) {
        return sycl::float2{i, i};
    } // this replication makes code cleaner later.
};
template<> struct packing<char> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = char;
    using packed_type = sycl::char2;
    static inline constexpr sycl::char2 pack(const char &i) {
        return sycl::char2{i, i};
    } // this replication makes code cleaner later.
};
template <> struct packing<sycl::char2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = char;
    using packed_type = sycl::char2;
    static inline constexpr sycl::char2 pack(const char &i) {
        return sycl::char2{i, i};
    } // this replication makes code cleaner later.
};
template<> struct packing<int> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = sycl::int2;
    static inline constexpr sycl::int2 pack(const int &i) {
        return sycl::int2{i, i};
    } // this replication makes code cleaner later.
};
template <> struct packing<sycl::int2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = sycl::int2;
    static inline constexpr sycl::int2 pack(const int &i) {
        return sycl::int2{i, i};
    } // this replication makes code cleaner later.
};
struct uint64_2 { uint64_t x, y; };
template<> struct packing<uint64_t> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<uint64_2> {
    static inline constexpr int num() { return 2; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template <> struct packing<sycl::float4> {
    static inline constexpr int num() { return 4; }
};
template <> struct packing<sycl::int4> {
    static inline constexpr int num() { return 4; }
};
#ifdef KITTENS_HOPPER
template<> struct packing<fp8e4m3> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e4m3_4> {
    static inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e5m2> {
    static inline constexpr int num() { return 1; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};
template<> struct packing<fp8e5m2_4> {
    static inline constexpr int num() { return 4; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};
#endif


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
    static inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static inline float convert(const bf16 & u) {
        return sycl::ext::intel::math::bfloat162float(u);
    }
};
template<> struct convertor<bf16, float> {
    static inline bf16 convert(const float & u) {
        return sycl::ext::intel::math::float2bfloat16_rn(u);
    }
};
template <> struct convertor<sycl::float2, bf16_2> {
    static inline sycl::float2 convert(const bf16_2 &u) {
        return sycl::float2(sycl::ext::intel::math::bfloat162float(u.x()),
                            sycl::ext::intel::math::bfloat162float(u.y()));
    }
};
template <> struct convertor<bf16_2, sycl::float2> {
    static inline bf16_2 convert(const sycl::float2 &u) {
        return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(u.x(), u.y());
    }
};
template <> struct convertor<float, sycl::half> {
    static inline float convert(const sycl::half &u) {
        return sycl::ext::intel::math::half2float(u);
    }
};
template <> struct convertor<sycl::half, float> {
    static inline sycl::half convert(const float &u) {
        return sycl::ext::intel::math::float2half_rn(u);
    }
};
template <> struct convertor<sycl::float2, half_2> {
    static inline sycl::float2 convert(const half_2 &u) {
        return sycl::float2(sycl::ext::intel::math::half2float(u.x()),
                            sycl::ext::intel::math::half2float(u.y()));
    }
};
template <> struct convertor<half_2, sycl::float2> {
    static inline half_2 convert(const sycl::float2 &u) {
        return sycl::half2(sycl::ext::intel::math::float2half_rn(u.x()),
                           sycl::ext::intel::math::float2half_rn(u.y()));
    }
};
template <> struct convertor<bf16, sycl::half> {
    static inline bf16 convert(const sycl::half &u) {
        return sycl::ext::intel::math::float2bfloat16_rn(
            sycl::ext::intel::math::half2float(u));
    }
};
template <> struct convertor<sycl::half, bf16> {
    static inline sycl::half convert(const bf16 &u) {
        return sycl::ext::intel::math::float2half_rn(
            sycl::ext::intel::math::bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static inline bf16_2 convert(const half_2 & u) {
        return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(
            sycl::float2(sycl::ext::intel::math::half2float(u.x()),
                         sycl::ext::intel::math::half2float(u.y()))
                .x(),
            sycl::float2(sycl::ext::intel::math::half2float(u.x()),
                         sycl::ext::intel::math::half2float(u.y()))
                .y());
    }
};
template<> struct convertor<half_2, bf16_2> {
    static inline half_2 convert(const bf16_2 & u) {
        return sycl::half2(
            sycl::ext::intel::math::float2half_rn(
                sycl::float2(sycl::ext::intel::math::bfloat162float(u.x()),
                             sycl::ext::intel::math::bfloat162float(u.y()))
                    .x()),
            sycl::ext::intel::math::float2half_rn(
                sycl::float2(sycl::ext::intel::math::bfloat162float(u.x()),
                             sycl::ext::intel::math::bfloat162float(u.y()))
                    .y()));
    }
};
#ifdef KITTENS_HOPPER
// fp8e4m3
template <> struct convertor<fp8e4m3_4, sycl::float4> {
    static inline fp8e4m3_4 convert(const sycl::float4 &u) {
        return __nv_fp8x4_e4m3(u); 
    }
};
template <> struct convertor<sycl::float4, fp8e4m3_4> {
    static inline sycl::float4 convert(const fp8e4m3_4 &u) {
        __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<__nv_fp8x4_e4m3*>(&u));
        return sycl::float4(float(vals[0]), float(vals[1]), float(vals[2]),
                            float(vals[3]));
    }
};
template <> struct convertor<fp8e4m3_2, sycl::float2> {
    static inline fp8e4m3_2 convert(const sycl::float2 &u) {
        return __nv_fp8x2_e4m3(u); 
    }
};
template <> struct convertor<sycl::float2, fp8e4m3_2> {
    static inline sycl::float2 convert(const fp8e4m3_2 &u) {
        __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<__nv_fp8x2_e4m3*>(&u));
        return sycl::float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e4m3, float> {
    static inline fp8e4m3 convert(const float & u) {
        return __nv_fp8_e4m3(u);
    }
};
template<> struct convertor<float, fp8e4m3> {
    static inline float convert(const fp8e4m3 & u) {
        return float(u);
    }
};
template<> struct convertor<bf16_2, fp8e4m3_4> {
    static inline bf16_2 convert(const fp8e4m3_4 & u) {
        sycl::float4 f4 = convertor<sycl::float4, fp8e4m3_4>::convert(u);
        sycl::float2 f2 = sycl::float2(f4.x(), f4.y());
        return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(f2.x(), f2.y());
    }
};
template<> struct convertor<fp8e4m3_4, bf16_2> {
    static inline fp8e4m3_4 convert(const bf16_2 & u) {
        sycl::float2 f2 =
            sycl::float2(sycl::ext::intel::math::bfloat162float(u.x()),
                         sycl::ext::intel::math::bfloat162float(u.y()));
        sycl::float4 f4 = sycl::float4(f2.x(), f2.y(), 0.0f, 0.0f);
        return __nv_fp8x4_e4m3(f4);
    }
};
// fp8e5m2
template <> struct convertor<fp8e5m2_4, sycl::float4> {
    static inline fp8e5m2_4 convert(const sycl::float4 &u) {
        return __nv_fp8x4_e5m2(u); 
    }
};
template <> struct convertor<sycl::float4, fp8e5m2_4> {
    static inline sycl::float4 convert(const fp8e5m2_4 &u) {
        __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<__nv_fp8x4_e5m2*>(&u));
        return sycl::float4(float(vals[0]), float(vals[1]), float(vals[2]),
                            float(vals[3]));
    }
};
template <> struct convertor<fp8e5m2_2, sycl::float2> {
    static inline fp8e5m2_2 convert(const sycl::float2 &u) {
        return __nv_fp8x2_e5m2(u); 
    }
};
template <> struct convertor<sycl::float2, fp8e5m2_2> {
    static inline sycl::float2 convert(const fp8e5m2_2 &u) {
        __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<__nv_fp8x2_e5m2*>(&u));
        return sycl::float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e5m2, float> {
    static inline fp8e5m2 convert(const float & u) {
        return __nv_fp8_e5m2(u);
    }
};
template<> struct convertor<float, fp8e5m2> {
    static inline float convert(const fp8e5m2 & u) {
        return float(u);
    }
};
template<> struct convertor<bf16_2, fp8e5m2_4> {
    static inline bf16_2 convert(const fp8e5m2_4 & u) {
        sycl::float4 f4 = convertor<sycl::float4, fp8e5m2_4>::convert(u);
        sycl::float2 f2 = sycl::float2(f4.x(), f4.y());
        return sycl::vec<sycl::ext::oneapi::bfloat16, 2>(f2.x(), f2.y());
    }
};
template<> struct convertor<fp8e5m2_4, bf16_2> {
    static inline fp8e5m2_4 convert(const bf16_2 & u) {
        sycl::float2 f2 =
            sycl::float2(sycl::ext::intel::math::bfloat162float(u.x()),
                         sycl::ext::intel::math::bfloat162float(u.y()));
        sycl::float4 f4 = sycl::float4(f2.x(), f2.y(), 0.0f, 0.0f);
        return __nv_fp8x4_e5m2(f4);
    }
};
#endif
}
}
