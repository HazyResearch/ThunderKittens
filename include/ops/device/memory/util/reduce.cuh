/**
 * @file
 * @brief Implementations for multimem_reduce operations
 */

#pragma once

namespace kittens {

template<typename T> struct multimem_reduce {
    __device__ static inline void add(T *dst, T value);
    __device__ static inline void min(T *dst, T value);
    __device__ static inline void max(T *dst, T value);
};

template<> struct multimem_reduce<bf16> { // TODO: Clarify on why b16 instead of bf16 here
    __device__ static inline void add(bf16 *dst, bf16 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.b16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void min(bf16 *dst, bf16 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.b16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void max(bf16 *dst, bf16 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.b16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<half> {
    __device__ static inline void add(half *dst, half value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.f16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void min(half *dst, half value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.f16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void max(half *dst, half value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.f16 [%0], %1;"
            :
            : "l"(dst), "h"(*(uint16_t*)&value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<float> {
    __device__ static inline void add(float *dst, float value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.f32 [%0], %1;"
            :
            : "l"(dst), "f"(value)
            : "memory"
        );
    }
    __device__ static inline void min(float *dst, float value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.f32 [%0], %1;"
            :
            : "l"(dst), "f"(value)
            : "memory"
        );
    }
    __device__ static inline void max(float *dst, float value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.f32 [%0], %1;"
            :
            : "l"(dst), "f"(value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<bf16_2> {
    __device__ static inline void add(bf16_2 *dst, bf16_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.b16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void min(bf16_2 *dst, bf16_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.b16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void max(bf16_2 *dst, bf16_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.b16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<half_2> {
    __device__ static inline void add(half_2 *dst, half_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.f16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void min(half_2 *dst, half_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.f16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
    __device__ static inline void max(half_2 *dst, half_2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.f16 [%0], %1;"
            :
            : "l"(dst), "r"(*(uint32_t*)&value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<float2> {
    __device__ static inline void add(float2 *dst, float2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y)
            : "memory"
        );
    }
    __device__ static inline void min(float2 *dst, float2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y)
            : "memory"
        );
    }
    __device__ static inline void max(float2 *dst, float2 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<float4> {
    __device__ static inline void add(float4 *dst, float4 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
            : "memory"
        );
    }
    __device__ static inline void min(float4 *dst, float4 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
            : "memory"
        );
    }
    __device__ static inline void max(float4 *dst, float4 value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
            : "memory"
        );
    }
};

} // namespace kittens