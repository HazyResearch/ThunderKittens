/**
 * @file
 * @brief Implementations for multimem_ld_reduce operations
 */

#pragma once

namespace kittens {

// NOTE: Currently this code is old, all reduction ops are in reduce.cuh
// Keeping so old profiling code can still be ran for now

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

template<typename T> struct multimem_reduce {
    __device__ static inline void add(T *dst, T value);
    __device__ static inline void min(T *dst, T value);
    __device__ static inline void max(T *dst, T value);
};

template<> struct multimem_reduce<bf16> {
    __device__ static inline void add(bf16 *dst, bf16 *value) {
        unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
                                __bfloat16_as_ushort(value[1]);
        unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
                                __bfloat16_as_ushort(value[3]);
        unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
                                __bfloat16_as_ushort(value[5]);
        unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
                                __bfloat16_as_ushort(value[7]);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
    __device__ static inline void min(bf16 *dst, bf16 *value) {
        unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
                                __bfloat16_as_ushort(value[1]);
        unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
                                __bfloat16_as_ushort(value[3]);
        unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
                                __bfloat16_as_ushort(value[5]);
        unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
                                __bfloat16_as_ushort(value[7]);
        asm volatile(
            "multimem.red.relaxed.sys.global.min.v4.bf16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    } 
    __device__ static inline void max(bf16 *dst, bf16 *value) {
        unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
                                __bfloat16_as_ushort(value[1]);
        unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
                                __bfloat16_as_ushort(value[3]);
        unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
                                __bfloat16_as_ushort(value[5]);
        unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
                                __bfloat16_as_ushort(value[7]);
        asm volatile(
            "multimem.red.relaxed.sys.global.max.v4.bf16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<half> {
    __device__ static inline void add(half *dst, half *value) {
        unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
                                __half_as_ushort(value[1]);
        unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
                                __half_as_ushort(value[3]);
        unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
                                __half_as_ushort(value[5]);
        unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
                                __half_as_ushort(value[7]);

        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
    __device__ static inline void min(half *dst, half *value) {
        unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
                                __half_as_ushort(value[1]);
        unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
                                __half_as_ushort(value[3]);
        unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
                                __half_as_ushort(value[5]);
        unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
                                __half_as_ushort(value[7]);

        asm volatile(
            "multimem.red.relaxed.sys.global.min.v4.f16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
    __device__ static inline void max(half *dst, half *value) {
        unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
                                __half_as_ushort(value[1]);
        unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
                                __half_as_ushort(value[3]);
        unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
                                __half_as_ushort(value[5]);
        unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
                                __half_as_ushort(value[7]);

        asm volatile(
            "multimem.red.relaxed.sys.global.max.v4.f16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<float> {
    __device__ static inline void add(float *dst, float *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
            : "memory"
        );
    }
    __device__ static inline void min(float *dst, float *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.min.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
            : "memory"
        );
    }
    __device__ static inline void max(float *dst, float *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.max.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
            : "memory"
        );
    }
};

} // namespace kittens