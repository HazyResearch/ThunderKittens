#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/math.hpp>
/**
 * @file
 * @brief Implementations for multimem.red and multimem.ld_reduce operations
 */

#pragma once

namespace kittens {

enum class ReduceOp {
    ADD,
    MIN,
    MAX
};


template<typename T, ReduceOp Op> 
struct multimem_reduce_op {
    static inline void apply(T *dst, T *src);
    static inline void apply_vec(T *dst, T *src);
};

// For floating point types, only ADD is supported for .red 
template<>
struct multimem_reduce_op<bf16, ReduceOp::ADD> {
    static inline void apply_vec(bf16* dst, bf16* src) {
        unsigned int packed1 =
            (sycl::ext::intel::math::bfloat16_as_ushort(src[1]) << 16) |
            sycl::ext::intel::math::bfloat16_as_ushort(src[0]);
        unsigned int packed2 =
            (sycl::ext::intel::math::bfloat16_as_ushort(src[3]) << 16) |
            sycl::ext::intel::math::bfloat16_as_ushort(src[2]);
        unsigned int packed3 =
            (sycl::ext::intel::math::bfloat16_as_ushort(src[5]) << 16) |
            sycl::ext::intel::math::bfloat16_as_ushort(src[4]);
        unsigned int packed4 =
            (sycl::ext::intel::math::bfloat16_as_ushort(src[7]) << 16) |
            sycl::ext::intel::math::bfloat16_as_ushort(src[6]);
        /*
        DPCT1053:51: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1, "
                     "%2, %3, %4};"
                     :
                     : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3),
                       "r"(packed4)
                     : "memory");
    }
};

template <> struct multimem_reduce_op<sycl::half, ReduceOp::ADD> {
    static inline void apply_vec(sycl::half *dst, sycl::half *src) {
        unsigned int packed1 =
            (sycl::bit_cast<unsigned short, sycl::half>(src[1]) << 16) |
            sycl::bit_cast<unsigned short, sycl::half>(src[0]);
        unsigned int packed2 =
            (sycl::bit_cast<unsigned short, sycl::half>(src[3]) << 16) |
            sycl::bit_cast<unsigned short, sycl::half>(src[2]);
        unsigned int packed3 =
            (sycl::bit_cast<unsigned short, sycl::half>(src[5]) << 16) |
            sycl::bit_cast<unsigned short, sycl::half>(src[4]);
        unsigned int packed4 =
            (sycl::bit_cast<unsigned short, sycl::half>(src[7]) << 16) |
            sycl::bit_cast<unsigned short, sycl::half>(src[6]);
        /*
        DPCT1053:52: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1, "
                     "%2, %3, %4};"
                     :
                     : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3),
                       "r"(packed4)
                     : "memory");
    }
};

template<>
struct multimem_reduce_op<float, ReduceOp::ADD> {
    static inline void apply(float *dst, float *src) {
        /*
        DPCT1053:53: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"
                     :
                     : "l"(dst), "f"(src[0])
                     : "memory");
    }
    static inline void apply_vec(float* dst, float* src) {
        /*
        DPCT1053:54: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
            : "memory");
    }
};

template<> 
struct multimem_reduce_op<bf16_2, ReduceOp::ADD> {
    static inline void apply(bf16_2 *dst, bf16_2 *src) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(src);
        /*
        DPCT1053:55: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.red.relaxed.sys.global.add.bf16x2 [%0], %1;"
                     :
                     : "l"(dst), "r"(packed_value)
                     : "memory");
    }
};

template<> 
struct multimem_reduce_op<half_2, ReduceOp::ADD> {
    static inline void apply(half_2 *dst, half_2 *src) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(src);
        /*
        DPCT1053:56: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.red.relaxed.sys.global.add.f16x2 [%0], %1;"
                     :
                     : "l"(dst), "r"(packed_value)
                     : "memory");
    }
};

template <> struct multimem_reduce_op<sycl::float2, ReduceOp::ADD> {
    static inline void apply(sycl::float2 *dst, sycl::float2 *src) {
        /*
        DPCT1053:57: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(src->x()), "f"(src->y())
            : "memory");
    }
};


template<typename T, ReduceOp Op> 
struct multimem_ld_reduce_op {
    static inline void apply(T *dst, T *src);
    static inline void apply_vec(T *dst, T *src);
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::ADD> {
    static inline void apply_vec(sycl::float4 *dst, bf16 *src) {
        /*
        DPCT1053:58: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=f"(dst->x()), "=f"(dst->y()), "=f"(dst->z()),
                       "=f"(dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::MIN> {
    static inline void apply_vec(sycl::float4 *dst, bf16 *src) {
        /*
        DPCT1053:59: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min.v4.bf16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=f"(dst->x()), "=f"(dst->y()), "=f"(dst->z()),
                       "=f"(dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::MAX> {
    static inline void apply_vec(sycl::float4 *dst, bf16 *src) {
        /*
        DPCT1053:60: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=f"(dst->x()), "=f"(dst->y()), "=f"(dst->z()),
                       "=f"(dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template <> struct multimem_ld_reduce_op<sycl::half, ReduceOp::ADD> {
    static inline void apply_vec(sycl::float4 *dst, sycl::half *src) {
        sycl::int4 *_dst = reinterpret_cast<sycl::int4 *>(
            dst); // keep float4 as input for consistency
        /*
        DPCT1053:61: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=r"(_dst->x()), "=r"(_dst->y()), "=r"(_dst->z()),
                       "=r"(_dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template <> struct multimem_ld_reduce_op<sycl::half, ReduceOp::MIN> {
    static inline void apply_vec(sycl::float4 *dst, sycl::half *src) {
        sycl::int4 *_dst = reinterpret_cast<sycl::int4 *>(
            dst); // keep float4 as input for consistency
        /*
        DPCT1053:62: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min.v4.f16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=r"(_dst->x()), "=r"(_dst->y()), "=r"(_dst->z()),
                       "=r"(_dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template <> struct multimem_ld_reduce_op<sycl::half, ReduceOp::MAX> {
    static inline void apply_vec(sycl::float4 *dst, sycl::half *src) {
        sycl::int4 *_dst = reinterpret_cast<sycl::int4 *>(
            dst); // keep float4 as input for consistency
        /*
        DPCT1053:63: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.v4.f16x2 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=r"(_dst->x()), "=r"(_dst->y()), "=r"(_dst->z()),
                       "=r"(_dst->w())
                     : "l"(src)
                     : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<float, ReduceOp::ADD> {
    static inline void apply_vec(sycl::float4 *dst, float *src) {
        /*
        DPCT1053:64: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, "
                     "%1, %2, %3}, [%4];"
                     : "=f"(dst->x()), "=f"(dst->y()), "=f"(dst->z()),
                       "=f"(dst->w())
                     : "l"(src)
                     : "memory");
    }
};

// MIN/MAX ops are NOT supported on float32

template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::ADD> {
    static inline void apply(bf16_2* dst, bf16_2 *src) {
        /*
        DPCT1053:65: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int *>(dst))
            : "l"(src)
            : "memory");
    }
};
template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::MIN> {
    static inline void apply(bf16_2* dst, bf16_2 *src) {
        /*
        DPCT1053:66: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int *>(dst))
            : "l"(src)
            : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::MAX> {
    static inline void apply(bf16_2* dst, bf16_2 *src) {
        /*
        DPCT1053:67: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int *>(dst))
            : "l"(src)
            : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::ADD> {
    static inline void apply(half_2* dst, half_2 *src) {
        /*
        DPCT1053:68: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];"
                     : "=r"(*reinterpret_cast<unsigned int *>(dst))
                     : "l"(src)
                     : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::MIN> {
    static inline void apply(half_2* dst, half_2 *src) {
        /*
        DPCT1053:69: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min.f16x2 %0, [%1];"
                     : "=r"(*reinterpret_cast<unsigned int *>(dst))
                     : "l"(src)
                     : "memory");
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::MAX> {
    static inline void apply(half_2* dst, half_2 *src) {
        /*
        DPCT1053:70: Migration of device assembly code is not supported.
        */
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.f16x2 %0, [%1];"
                     : "=r"(*reinterpret_cast<unsigned int *>(dst))
                     : "l"(src)
                     : "memory");
    }
};

template <> struct multimem_ld_reduce_op<sycl::float2, ReduceOp::ADD> {
    static inline void apply(sycl::float2 *dst, sycl::float2 *src) {
        /*
        DPCT1053:71: Migration of device assembly code is not supported.
        */
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst->x()), "=f"(dst->y())
            : "l"(src)
            : "memory");
    }
};

} // namespace kittens