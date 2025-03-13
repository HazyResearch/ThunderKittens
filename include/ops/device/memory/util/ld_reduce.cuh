/**
 * @file
 * @brief Implementations for multimem_ld_reduce operations
 */

#pragma once

namespace kittens {

template<typename T> struct multimem_ld_reduce {
    __device__ static inline void add(T &dst, T *src);
    __device__ static inline void min(T &dst, T *src);
    __device__ static inline void max(T &dst, T *src);
};

template<> struct multimem_ld_reduce<bf16> { // TODO: Clarify on why b16 instead of bf16 here
    __device__ static inline void add(bf16 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.b16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(bf16 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.b16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(bf16 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.b16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<half> {
    __device__ static inline void add(half &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.f16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(half &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.f16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(half &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.f16 %0, [%1];"
            : "=h"(*(uint16_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<float> {
    __device__ static inline void add(float &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
            : "=f"(dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(float &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.f32 %0, [%1];"
            : "=f"(dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.f32 %0, [%1];"
            : "=f"(dst)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<bf16_2> { // TODO: Check if should use .v2 instead
    __device__ static inline void add(bf16_2 &dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.b16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(bf16_2 &dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.b16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(bf16_2 &dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.b16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<half_2> { // TODO: Check if should use .v2 instead
    __device__ static inline void add(half_2 &dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.f16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(half_2 &dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.f16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(half_2 &dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.f16 %0, [%1];"
            : "=r"(*(uint32_t*)&dst)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<float2> {
    __device__ static inline void add(float2 &dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst.x), "=f"(dst.y)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(float2 &dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst.x), "=f"(dst.y)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float2 &dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst.x), "=f"(dst.y)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<float4> {
    __device__ static inline void add(float4 &dst, float4 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(float4 &dst, float4 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float4 &dst, float4 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
};

} // namespace kittens