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

template<> struct multimem_ld_reduce<bf16> {
    __device__ static inline void add(float4 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" 
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src) 
            : "memory"
        );
    }
    __device__ static inline void min(float4 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float4 &dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<half> {
    __device__ static inline void add(float4 &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(float4 &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float4 &dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
};

template<> struct multimem_ld_reduce<float> {
    __device__ static inline void add(float4 &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void min(float4 &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
    __device__ static inline void max(float4 &dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
            : "l"(src)
            : "memory"
        );
    }
};
} // namespace kittens