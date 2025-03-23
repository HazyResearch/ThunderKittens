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

template<> struct multimem_reduce<bf16_2> {
    __device__ static inline void add(bf16_2 *dst, bf16_2 *value) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(value);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.bf16x2 [%0], %1;"
            :
            : "l"(dst), "r"(packed_value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<half_2> {
    __device__ static inline void add(half_2 *dst, half_2 *value) {
        unsigned int packed_value = *reinterpret_cast<const unsigned int*>(value);
        asm volatile(
            "multimem.red.relaxed.sys.global.add.f16x2 [%0], %1;"
            :
            : "l"(dst), "r"(packed_value)
            : "memory"
        );
    }
};

template<> struct multimem_reduce<float2> {
    __device__ static inline void add(float2 *dst, float2 *value) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1, %2};"
            :
            : "l"(dst), "f"(value->x), "f"(value->y)
            : "memory"
        );
    }
};

// template<> struct multimem_reduce<bf16> {
//     __device__ static inline void add(bf16 *dst, bf16 *value) {
//         unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
//                                 __bfloat16_as_ushort(value[1]);
//         unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
//                                 __bfloat16_as_ushort(value[3]);
//         unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
//                                 __bfloat16_as_ushort(value[5]);
//         unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
//                                 __bfloat16_as_ushort(value[7]);
//         asm volatile(
//             "multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     }
//     __device__ static inline void min(bf16 *dst, bf16 *value) {
//         unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
//                                 __bfloat16_as_ushort(value[1]);
//         unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
//                                 __bfloat16_as_ushort(value[3]);
//         unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
//                                 __bfloat16_as_ushort(value[5]);
//         unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
//                                 __bfloat16_as_ushort(value[7]);
//         asm volatile(
//             "multimem.red.relaxed.sys.global.min.v4.bf16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     } 
//     __device__ static inline void max(bf16 *dst, bf16 *value) {
//         unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
//                                 __bfloat16_as_ushort(value[1]);
//         unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
//                                 __bfloat16_as_ushort(value[3]);
//         unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
//                                 __bfloat16_as_ushort(value[5]);
//         unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
//                                 __bfloat16_as_ushort(value[7]);
//         asm volatile(
//             "multimem.red.relaxed.sys.global.max.v4.bf16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     }
// };

// template<> struct multimem_reduce<half> {
//     __device__ static inline void add(half *dst, half *value) {
//         unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
//                                 __half_as_ushort(value[1]);
//         unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
//                                 __half_as_ushort(value[3]);
//         unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
//                                 __half_as_ushort(value[5]);
//         unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
//                                 __half_as_ushort(value[7]);

//         asm volatile(
//             "multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     }
//     __device__ static inline void min(half *dst, half *value) {
//         unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
//                                 __half_as_ushort(value[1]);
//         unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
//                                 __half_as_ushort(value[3]);
//         unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
//                                 __half_as_ushort(value[5]);
//         unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
//                                 __half_as_ushort(value[7]);

//         asm volatile(
//             "multimem.red.relaxed.sys.global.min.v4.f16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     }
//     __device__ static inline void max(half *dst, half *value) {
//         unsigned int packed1 = (__half_as_ushort(value[0]) << 16) |
//                                 __half_as_ushort(value[1]);
//         unsigned int packed2 = (__half_as_ushort(value[2]) << 16) |
//                                 __half_as_ushort(value[3]);
//         unsigned int packed3 = (__half_as_ushort(value[4]) << 16) |
//                                 __half_as_ushort(value[5]);
//         unsigned int packed4 = (__half_as_ushort(value[6]) << 16) |
//                                 __half_as_ushort(value[7]);

//         asm volatile(
//             "multimem.red.relaxed.sys.global.max.v4.f16x2 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
//             : "memory"
//         );
//     }
// };

// template<> struct multimem_reduce<float> {
//     __device__ static inline void add(float *dst, float *value) {
//         asm volatile(
//             "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
//             : "memory"
//         );
//     }
//     __device__ static inline void min(float *dst, float *value) {
//         asm volatile(
//             "multimem.red.relaxed.sys.global.min.v4.f32 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
//             : "memory"
//         );
//     }
//     __device__ static inline void max(float *dst, float *value) {
//         asm volatile(
//             "multimem.red.relaxed.sys.global.max.v4.f32 [%0], {%1, %2, %3, %4};"
//             :
//             : "l"(dst), "f"(value[0]), "f"(value[1]), "f"(value[2]), "f"(value[3])
//             : "memory"
//         );
//     }
// };

} // namespace kittens