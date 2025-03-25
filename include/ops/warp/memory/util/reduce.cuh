/**
 * @file
 * @brief Implementations for multimem_reduce operations
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
    __device__ static inline void apply(T *dst, T *src);
    __device__ static inline void apply_vec(T *dst, T *src);
};

// For floating point types, only ADD is supported for .red 
template<>
struct multimem_reduce_op<bf16, ReduceOp::ADD> {
    __device__ static inline void apply_vec(bf16* dst, bf16* src) {
        unsigned int packed1 = (__bfloat16_as_ushort(src[0]) << 16) | 
                                __bfloat16_as_ushort(src[1]);
        unsigned int packed2 = (__bfloat16_as_ushort(src[2]) << 16) | 
                                __bfloat16_as_ushort(src[3]);
        unsigned int packed3 = (__bfloat16_as_ushort(src[4]) << 16) |
                                __bfloat16_as_ushort(src[5]);
        unsigned int packed4 = (__bfloat16_as_ushort(src[6]) << 16) |
                                __bfloat16_as_ushort(src[7]);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
};

template<>
struct multimem_reduce_op<half, ReduceOp::ADD> {
    __device__ static inline void apply_vec(half* dst, half* src) {
        unsigned int packed1 = (__half_as_ushort(src[0]) << 16) |
                                __half_as_ushort(src[1]);
        unsigned int packed2 = (__half_as_ushort(src[2]) << 16) |
                                __half_as_ushort(src[3]);
        unsigned int packed3 = (__half_as_ushort(src[4]) << 16) |
                                __half_as_ushort(src[5]);
        unsigned int packed4 = (__half_as_ushort(src[6]) << 16) |
                                __half_as_ushort(src[7]);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
            : "memory"
        );
    }
};

template<>
struct multimem_reduce_op<float, ReduceOp::ADD> {
    __device__ static inline void apply_vec(float* dst, float* src) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(dst), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
            : "memory"
        );
    }
};

template<> 
struct multimem_reduce_op<bf16_2, ReduceOp::ADD> {
    __device__ static inline void apply(bf16_2 *dst, bf16_2 *src) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(src);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.bf16x2 [%0], %1;"
            :
            : "l"(dst), "r"(packed_value)
            : "memory"
        );
    }
};

template<> 
struct multimem_reduce_op<half_2, ReduceOp::ADD> {
    __device__ static inline void apply(half_2 *dst, half_2 *src) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(src);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.f16x2 [%0], %1;"
            :
            : "l"(dst), "r"(packed_value)
            : "memory"
        );
    }
};

template<> 
struct multimem_reduce_op<float2, ReduceOp::ADD> {
    __device__ static inline void apply(float2 *dst, float2 *src) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(src->x), "f"(src->y)
            : "memory"
        );
    }
};


template<typename T, ReduceOp Op> 
struct multimem_ld_reduce_op {
    __device__ static inline void apply(T *dst, T *src);
    __device__ static inline void apply_vec(T *dst, T *src);
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::ADD> {
    __device__ static inline void apply_vec(float4 *dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::MIN> {
    __device__ static inline void apply_vec(float4 *dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<bf16, ReduceOp::MAX> {
    __device__ static inline void apply_vec(float4 *dst, bf16 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half, ReduceOp::ADD> {
    __device__ static inline void apply_vec(float4 *dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half, ReduceOp::MIN> {
    __device__ static inline void apply_vec(float4 *dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half, ReduceOp::MAX> {
    __device__ static inline void apply_vec(float4 *dst, half *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.f16x2 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float, ReduceOp::ADD> {
    __device__ static inline void apply_vec(float4 *dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float, ReduceOp::MIN> {
    __device__ static inline void apply_vec(float4 *dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::ADD> {
    __device__ static inline void apply(bf16_2* dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};
template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::MIN> {
    __device__ static inline void apply(bf16_2* dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<bf16_2, ReduceOp::MAX> {
    __device__ static inline void apply(bf16_2* dst, bf16_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.bf16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::ADD> {
    __device__ static inline void apply(half_2* dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::MIN> {
    __device__ static inline void apply(half_2* dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.f16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<half_2, ReduceOp::MAX> {
    __device__ static inline void apply(half_2* dst, half_2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.f16x2 %0, [%1];"
            : "=r"(*reinterpret_cast<unsigned int*>(dst))
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float2, ReduceOp::ADD> {
    __device__ static inline void apply(float2* dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst->x), "=f"(dst->y)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float2, ReduceOp::MIN> {
    __device__ static inline void apply(float2* dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.min.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst->x), "=f"(dst->y)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float2, ReduceOp::MAX> {
    __device__ static inline void apply(float2* dst, float2 *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v2.f32 {%0, %1}, [%2];"
            : "=f"(dst->x), "=f"(dst->y)
            : "l"(src)
            : "memory"
        );
    }
};

template<>
struct multimem_ld_reduce_op<float, ReduceOp::MAX> {
    __device__ static inline void apply_vec(float4 *dst, float *src) {
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.max.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(dst->x), "=f"(dst->y), "=f"(dst->z), "=f"(dst->w)
            : "l"(src)
            : "memory"
        );
    }
};

} // namespace kittens