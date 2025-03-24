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
    __device__ static inline void apply(T *dst, T *value);
};

// For floating point types, only ADD is supported for .red 
template<> 
struct multimem_reduce_op<float, ReduceOp::ADD> {
    __device__ static inline void apply(float *dst, float *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], %1;"
            :
            : "l"(dst), "f"(*value)
            : "memory"
        );
    }
};
template<> 
struct multimem_reduce_op<bf16_2, ReduceOp::ADD> {
    __device__ static inline void apply(bf16_2 *dst, bf16_2 *value) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(value);
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
    __device__ static inline void apply(half_2 *dst, half_2 *value) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(value);
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
    __device__ static inline void apply(float2 *dst, float2 *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(value->x), "f"(value->y)
            : "memory"
        );
    }
};


template<typename T, ReduceOp Op> 
struct multimem_ld_reduce_op {
    __device__ static inline void apply_vec(T *dst, T *value);
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