/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in tensor memory.
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"
#include "../util/sync.cuh"

namespace kittens {
namespace detail {
namespace tcgen05 {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor
template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
__device__ static inline constexpr uint32_t instruction_descriptor() {
    uint32_t desc = 0;
    if constexpr (std::is_same_v<AB, half> || std::is_same_v<AB, bf16>) { // kind::f16
        // either accumulate to float, or the input is half and the output is half
        static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, half>) {
            desc |= 0b000 << 7;  // 16-bit A input type as FP16
            desc |= 0b000 << 10; // 16-bit B input type as FP16
        } else if constexpr (std::is_same_v<AB, bf16>) {
            desc |= 0b001 << 7;  // 16-bit A input type as BF16
            desc |= 0b001 << 10; // 16-bit B input type as BF16
        }
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    } else if constexpr (std::is_same_v<AB, int8> || std::is_same_v<AB, uint8>) { // kind::i8
        static_assert(std::is_same_v<D, int>, "Integer MMA must accumulate to int32.");
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b1       << 3;  // saturate output
        desc |= 0b10      << 4;  // D matrix is S32 (only option)
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, uint8>) {
            desc |= 0b000 << 7;  // 8-bit A input type as unsigned
            desc |= 0b000 << 10; // 8-bit B input type as unsigned
        } else if constexpr (std::is_same_v<AB, int8>) {
            desc |= 0b001 << 7;  // 8-bit A input type as signed
            desc |= 0b001 << 10; // 8-bit B input type as signed
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type.");
        }
        static_assert(!neg, "Negation is not supported for integer matrix multiplies.");
        desc |= 0b0       << 13; // Don't negate A matrix
        desc |= 0b0       << 14; // Don't negate B matrix
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    } else if constexpr (std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp8e5m2> || std::is_same_v<AB, fp4e2m1_2>) { // kind::f8f6f4
        static_assert(std::is_same_v<D, float> || std::is_same_v<D, half>); // FP8/6/4 has to accumulate to float or half
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e2m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e2m1
        }
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    } else {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
    }
    return desc;
};

template<typename D, typename AB, typename SAB, int M, int N, bool neg=false, int scale_factor_id=0, int mma_k=64>
__device__ static inline constexpr uint32_t instruction_descriptor() {
    // Only supported types are MXFP8 and NVFP4
    static_assert(std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp4e2m1_2>, "AB must be fp8e4m3 for f4e2m1");
    static_assert(std::is_same_v<SAB, fp8e4m3> || std::is_same_v<SAB, fp8e8m0>, "SAB must be either fp8e4m3 or fp8e8m0");
    // K=96 is a packed-FP4 (mxf4nvf4) mode; the hardware allows either scale type
    // (block32/E8M0 == MXFP4, block16/E4M3 == NVFP4). SAB is already constrained to
    // {E4M3,E8M0} above, so only the data type and K need checking here.
    static_assert(
        mma_k == 64 || (std::is_same_v<AB, fp4e2m1_2> && mma_k == 96),
        "K96 block-scaled MMA is only supported for packed NVFP4 (E4M3 or E8M0 scales).");
    constexpr int scale_type = std::is_same_v<SAB, fp8e4m3> ? 0 : std::is_same_v<SAB, fp8e8m0> ? 1 : -1;

    uint32_t desc = 0;
    desc |= 0b00 << 0; // SBZ
    desc |= 0b0 << 2; // dense
    desc |= 0b0 << 3; // SBZ
    desc |= scale_factor_id << 4; // Matrix B scale Factor ID (0, 1, 2, 3 for MXFP8; 0, 2 for NVFP4)
    desc |= 0b0 << 6; // SBZ
    if constexpr (std::is_same_v<AB, fp8e4m3>) { // MXFP8
        desc |= (0b000 << 7); // Matrix A is E4M3
        desc |= (0b000 << 10); // Matrix B is E4M3
    } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) { // NVFP4
        desc |= 0b001 << 7; // Matrix A is E2M1
        desc |= 0b01 << 10; // Matrix B is E2M1
        desc |= 0b0 << 12;  // SBZ
    } else {
        static_assert(sizeof(AB) == 999, "Invalid AB type.");
    }
    if constexpr (neg) {
        desc |= 0b1 << 13; // Do negate A matrix
    }
    else {
        desc |= 0b0 << 13; // Don't negate A matrix
    }
    desc |= 0b0 << 14; // Don't negate B matrix (in all cases)
    desc |= 0b0 << 15; // Don't transpose A (in all cases)
    desc |= 0b0 << 16; // Don't transpose B (in all cases)
    desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
    desc |= scale_type   << 23; // Scale type (0 is ue4m3, 1 is ue8m0)
    desc |= 0b000 << 24; // SBZ
    desc |= (M >> 7) << 27; // A matrix has dimension M, encoded
    desc |= scale_factor_id  << 29; // Matrix A scale Factor ID (0, 1, 2, 3 for MXFP8; 0, 2 for NVFP4)
    desc |= (mma_k == 96 ? 1u : 0u) << 31; // K dimension (NVFP4: 0 is K=64, 1 is K=96; MXFP8: no choice, SBZ, K is always 32)

    return desc;
}

template<typename AB, int acc, int ncta=1>
__device__ static inline void tt_st(uint32_t d_tt_addr, uint32_t a_tt_addr, uint64_t b_desc, uint32_t idesc) {
    if constexpr (std::is_same_v<AB, int8> || std::is_same_v<AB, uint8>) {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else if constexpr (std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp8e5m2> || std::is_same_v<AB, fp4e2m1_2>) {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

template<typename AB, int acc, int ncta=1>
__device__ static inline void st_st(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    if constexpr (std::is_same_v<AB, int8> || std::is_same_v<AB, uint8>) {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else if constexpr (std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp8e5m2> || std::is_same_v<AB, fp4e2m1_2>) {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

template<typename AB, typename SAB, int acc, int ncta=1, int block_size=16>
__device__ static inline void st_st(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t sa_tt_addr, uint32_t sb_tt_addr, uint32_t idesc) {
    static_assert(std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp4e2m1_2>, "AB must be fp8e4m3 for f4e2m1");
    if constexpr (ncta == 1) {
        if constexpr (std::is_same_v<AB, fp8e4m3>) { // Block size is always 32; alias is 1X
            asm volatile(
                "{.reg .pred p;\n\t" \
                "setp.eq.u32 p, 1, %6;\n\t" \
                "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
            );
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            if constexpr (block_size == 32) { // E8M0 scale only
                asm volatile(
                    "{.reg .pred p;\n\t"
                    "setp.eq.u32 p, 1, %6;\n\t"
#if !defined(KITTENS_SM103) && (__CUDACC_VER_MAJOR__ > 12)
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;}\n"
#else
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;}\n"
#endif
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
            else { // E4M3 or E8M0 scale
                asm volatile( // block_size == 16 is an alias for scale_vec::4X
                "{.reg .pred p;\n\t"
                    "setp.eq.u32 p, 1, %6;\n\t"
#if !defined(KITTENS_SM103) && (__CUDACC_VER_MAJOR__ > 12)
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;}\n"
#else
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;}\n"
#endif
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type.");
        }
    }
    else {
        if constexpr (std::is_same_v<AB, fp8e4m3>) { // Block size is always 32; alias is 1X
            asm volatile(
                "{.reg .pred p;\n\t" \
                "setp.eq.u32 p, 1, %6;\n\t" \
                "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
            );
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            if constexpr (block_size == 32) { // E8M0 scale only
                asm volatile(
                "{.reg .pred p;\n\t"
                    "setp.eq.u32 p, 1, %6;\n\t"
#if !defined(KITTENS_SM103) && (__CUDACC_VER_MAJOR__ > 12)
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;}\n"
#else
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;}\n"
#endif
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
            else {  // E4M3 or E8M0 scale
                asm volatile( // block_size == 16 is an alias for scale_vec::4X
                "{.reg .pred p;\n\t"
                    "setp.eq.u32 p, 1, %6;\n\t"
#if !defined(KITTENS_SM103) && (__CUDACC_VER_MAJOR__ > 12)
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;}\n"
#else
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;}\n"
#endif
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type.");
        }
    }
}

template <int ncta>
__device__ static inline void commit(kittens::semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    if constexpr (ncta == 1) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
        ::  "l"(__cvta_generic_to_shared(&sem)));
    }
    else {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        ::  "l"(__cvta_generic_to_shared(&sem)), "h"(dst_cta_mask));
    }
}

} // namespace tcgen05
} // namespace detail

// Reduction dimension for non-mx formats
template<typename T_AB> constexpr int reduction_dimension = sizeof(T_AB) == 2 ? 16 : sizeof(T_AB) == 4 ? 8 : 32;

// TS matmul
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;

    // Do everything here.
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256)))); // output register is correctly sized

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    static_assert(N == D::cols); // output register is correctly sized

    constexpr int K = trans_a ? A::rows : A::cols;
    static_assert((trans_b ? B::rows : B::cols) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        // Half output with supported input types
        (std::is_same_v<T_D, half> && (
            std::is_same_v<T_AB, half> || 
            std::is_same_v<T_AB, fp8e4m3> || 
            std::is_same_v<T_AB, fp8e5m2> || 
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // Float output with supported input types  
        (std::is_same_v<T_D, float> && (
            std::is_same_v<T_AB, bf16> || 
            std::is_same_v<T_AB, half> ||
            std::is_same_v<T_AB, fp8e4m3> ||
            std::is_same_v<T_AB, fp8e5m2> ||
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // Int output with supported input types
        (std::is_same_v<T_D, int> && (
            std::is_same_v<T_AB, int8> ||
            std::is_same_v<T_AB, uint8>
        )),
        "Currently unsupported type combination for matrix multiply."
    );

    static_assert(trans_a == transpose::N, "A from TMEM can't be transposed in TS (tensor-shared) MMA.");

    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

    detail::tcgen05::template tt_st<T_AB, acc, ncta>(
        d.addr,
        a.template chunk_addr<trans_a>(0),
        b_desc.chunk_descriptor(0),
        idesc
    );

    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template tt_st<T_AB, 1, ncta>(
            d.addr,
            a.template chunk_addr<trans_a>(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);
}

// SS matmul
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;

    // Do everything here.
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256)))); // output register is correctly sized

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    static_assert(N == D::cols); // output register is correctly sized

    // constexpr int K = std::is_same_v<typename A::T, fp4e2m1_2> ? (trans_a ? A::rows : A::cols) * 2 : (trans_a ? A::rows : A::cols);
    constexpr int K = (trans_a ? A::rows : A::cols);
    static_assert((trans_b ? B::rows : B::cols) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        // Half output with supported input types
        (std::is_same_v<T_D, half> && (
            std::is_same_v<T_AB, half> || 
            std::is_same_v<T_AB, fp8e4m3> || 
            std::is_same_v<T_AB, fp8e5m2> || 
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // Float output with supported input types  
        (std::is_same_v<T_D, float> && (
            std::is_same_v<T_AB, bf16> || 
            std::is_same_v<T_AB, half> ||
            std::is_same_v<T_AB, fp8e4m3> ||
            std::is_same_v<T_AB, fp8e5m2> ||
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // Int output with supported input types
        (std::is_same_v<T_D, int> && (
            std::is_same_v<T_AB, int8> ||
            std::is_same_v<T_AB, uint8>
        )),
        "Currently unsupported type combination for matrix multiply."
    );

    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    
    detail::tcgen05::template st_st<T_AB, acc, ncta>(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        idesc
    );
    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template st_st<T_AB, 1, ncta>(
            d.addr,
            a_desc.chunk_descriptor(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);
}

// SS matmul with microscaling (MXFP8 and NVFP4)
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1, int mma_k=64>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {

    // Check that A and B are fp8e4m3 or fp4e2m1 and the scales match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.
    static_assert(
        (std::is_same_v<typename A::T, fp8e4m3> && std::is_same_v<typename B::T, fp8e4m3>) ||
        (std::is_same_v<typename A::T, fp4e2m1_2> && std::is_same_v<typename B::T, fp4e2m1_2>),
        "A and B must be fp8e4m3 or fp4e2m1_2"
    );
    static_assert(
        (std::is_same_v<typename A::T, fp8e4m3> && (
            std::is_same_v<typename SA::T, fp8e8m0> && std::is_same_v<typename SB::T, fp8e8m0>
        )) || 
        (std::is_same_v<typename A::T, fp4e2m1_2> && (
            (std::is_same_v<typename SA::T, fp8e8m0> && std::is_same_v<typename SB::T, fp8e8m0>) ||
            (std::is_same_v<typename SA::T, fp8e4m3> && std::is_same_v<typename SB::T, fp8e4m3>)
        )),
        "SAB must be fp8e8m0 for fp8e4m3 element type, or fp8e4m3 / fp8e8m0 for fp4e2m1_2 element type");
    // Only float32 accumulator is supported for microscaling formats
    static_assert(std::is_same_v<typename D::T, float>, "Only float32 accumulator is supported for microscaling formats");
    using T_AB = A::T;
    using T_SAB = SA::T;
    using T_D  = D::T;

    constexpr int block_size = std::is_same_v<typename SA::T, fp8e4m3> ? 16 : 32;
    constexpr int trans_b = 1 - n_trans_b;
#if !defined(KITTENS_SM103)
    static_assert(mma_k != 96, "NVFP4 K96 tcgen05 MMA is only supported on SM103.");
#endif
    static_assert(mma_k == 64 || mma_k == 96, "Microscaling MMA K dimension must be 64 or 96.");
    if constexpr (mma_k == 96) {
        static_assert(std::is_same_v<typename A::T, fp4e2m1_2>, "K96 microscaling MMA is only supported for packed FP4.");
        static_assert(
            (std::is_same_v<typename SA::T, fp8e8m0> && std::is_same_v<typename SB::T, fp8e8m0>) ||
            (std::is_same_v<typename SA::T, fp8e4m3> && std::is_same_v<typename SB::T, fp8e4m3>),
            "K96 expects matching E8M0 (MXFP4, block32) or E4M3 (NVFP4, block16) scale factors.");
        static_assert(ncta == 1 || ncta == 2, "K96 NVFP4 support expects cta_group::1 or cta_group::2.");
        static_assert(trans_a == transpose::N && trans_b == transpose::N, "K96 NVFP4 support currently expects ABt/K-major operands.");
    }

    // Matrix dimension calculations
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    constexpr int K = std::is_same_v<typename A::T, fp4e2m1_2> ? (trans_a ? A::rows : A::cols) * 2 : (trans_a ? A::rows : A::cols);
    constexpr bool is_k96 = std::is_same_v<typename A::T, fp4e2m1_2> && mma_k == 96;
    constexpr int red_dim = std::is_same_v<typename A::T, fp8e4m3> ? 32 : mma_k;
    static_assert(
        (is_k96 && K >= red_dim) ||
        (!is_k96 && K % red_dim == 0),
        "K dimension must be divisible by the selected tcgen05 MMA K dimension.");

    // M is 128 for 1 CTA, 128 or 256 for 2 CTAs
    static_assert(M == D::rows*ncta && ((ncta == 1 && M == 128) || (ncta == 2 && (M == 128 || M == 256))));

    // valid N are steps of 8 for 1 CTA, steps of 16 for 2 CTAs
    static_assert(N == D::cols && ((ncta == 1 && N%8 == 0) || (ncta == 2 && N%16 == 0)));

    // Get shared tile descriptors
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    // This is not neeeded if using mbarrier synchronization with acquire/release (default) semantics.
    // However, it must be used if manually storing to SMEM/TMEM with generic operations.
    // asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

    // Generate instruction descriptors
    constexpr uint32_t idescs[4] = {
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0, mma_k>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 1, mma_k>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 2, mma_k>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 3, mma_k>()
    };

    detail::tcgen05::template st_st<T_AB, T_SAB, acc, ncta, block_size>(
        d.addr,
        a_desc.template chunk_descriptor<mma_k>(0),
        b_desc.template chunk_descriptor<mma_k>(0),
        sa.addr,
        sb.addr,
        idescs[0]
    );

    // Offsets for moving the scales
    constexpr int N_offset = N / 32; // 8 if N=256
    constexpr int M_offset = M / 32 / ncta; // 4 if M=256
    if constexpr (is_k96) {
        static_assert(((K / red_dim) - 1) * M_offset < SA::cols / 4, "A scale tensor tile is too narrow for the K96 MMA count.");
        static_assert(((K / red_dim) - 1) * N_offset < SB::cols / 4, "B scale tensor tile is too narrow for the K96 MMA count.");
    }

    if constexpr (std::is_same_v<typename A::T, fp8e4m3>) { // FP8E4M3 + FP8E8M0 scale (MXFP8)
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.chunk_descriptor(i),
                b_desc.chunk_descriptor(i),
                sa.addr + (i >> 2) * M_offset, // i / 4
                sb.addr + (i >> 2) * N_offset, // i / 4
                idescs[i % 4]
            );
        }
    } else if constexpr (std::is_same_v<typename A::T, fp4e2m1_2> && block_size == 16) { // FP4E2M1 + FP8E4M3 scale (NVFP4)
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.template chunk_descriptor<mma_k>(i),
                b_desc.template chunk_descriptor<mma_k>(i),
                sa.addr + i * M_offset,
                sb.addr + i * N_offset,
                idescs[0] // SFID is always 0
            );
        }
    } else if constexpr (std::is_same_v<typename A::T, fp4e2m1_2> && block_size == 32) { // FP4E2M1 + FP8E8M0 scale
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.template chunk_descriptor<mma_k>(i),
                b_desc.template chunk_descriptor<mma_k>(i),
                // K64 pairs MMAs on one scale address and alternates SFID 0/2;
                // K96 uses SFID 0 and advances one aligned scale panel per MMA.
                sa.addr + (mma_k == 96 ? i * M_offset : (i >> 1) * M_offset),
                sb.addr + (mma_k == 96 ? i * N_offset : (i >> 1) * N_offset),
                (mma_k == 96 || !(i & 1)) ? idescs[0] : idescs[2]
            );
        }
    } else {
        static_assert(sizeof(T_AB) == 999, "Should not reach here.");
    }
}
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1, int mma_k=64>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    mma<trans_a, n_trans_b, D, A, B, SA, SB, acc, ncta, mma_k>(d, a, b, sa, sb);
    detail::tcgen05::commit<ncta>(sem, dst_cta_mask);
}

// Accumulator / numcta wrappers
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int mma_k=64>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int mma_k=64>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2, mma_k>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, 0, 1, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, 0, 1, mma_k>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0, mma_k>(d, a, b, sa, sb);
}

// Transpose wrappers
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 1, 96>(d, a, b, sa, sb, sem, dst_cta_mask);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 1, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 2, 96>(d, a, b, sa, sb, sem, dst_cta_mask);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 2, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0, 1, 96>(d, a, b, sa, sb, sem, dst_cta_mask);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0, 1, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0, 2, 96>(d, a, b, sa, sb, sem, dst_cta_mask);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0, 2, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}

// ---------------------------------------------------------------------------
// Single-chunk NVFP4 K96 microscaling MMA.
//
// Issues exactly one tcgen05 MMA for K96 chunk `chunk_idx` of the (full,
// contiguous) A/B shared tiles, reading the matching scale panel from TMEM.
// This mirrors one iteration of the K96 loop inside the core `mma`, but exposes
// it so the consumer can interleave per-chunk MMAs with finer-grained input
// readiness barriers (CuteDSL-style sub-tile staging). `init` selects whether
// this chunk initializes (acc=0) or accumulates (acc=1) into the accumulator.
// No commit is performed; call `mma_k96_commit` after the final chunk.
template<int trans_a, int n_trans_b, int ncta, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_k96_chunk(D &d, const A &a, const B &b, const SA &sa, const SB &sb, int chunk_idx, bool init) {
    using T_AB  = typename A::T;
    using T_SAB = typename SA::T;
    using T_D   = typename D::T;
    static_assert(std::is_same_v<T_AB, fp4e2m1_2>, "mma_k96_chunk is only for NVFP4 operands.");
    constexpr int block_size = std::is_same_v<typename SA::T, fp8e4m3> ? 16 : 32;
    constexpr int trans_b = 1 - n_trans_b;
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    constexpr int M_offset = M / 32 / ncta;
    constexpr int N_offset = N / 32;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    constexpr uint32_t idesc0 = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0, 96>();
    constexpr int sf_atoms = ((96 / block_size) + 3) / 4;
    const uint64_t ad = a_desc.template chunk_descriptor<96>(chunk_idx);
    const uint64_t bd = b_desc.template chunk_descriptor<96>(chunk_idx);
    if (init)
        detail::tcgen05::template st_st<T_AB, T_SAB, 0, ncta, block_size>(
            d.addr, ad, bd, sa.addr + chunk_idx * M_offset * sf_atoms, sb.addr + chunk_idx * N_offset * sf_atoms, idesc0);
    else
        detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
            d.addr, ad, bd, sa.addr + chunk_idx * M_offset * sf_atoms, sb.addr + chunk_idx * N_offset * sf_atoms, idesc0);
}
template<int ncta>
__device__ static inline void mma_k96_commit(semaphore &sem) {
    detail::tcgen05::commit<ncta>(sem);
}
// ABt, cta_group::2 convenience wrapper matching mm2_ABt_k96/mma2_ABt_k96.
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k96_chunk(D &d, const A &a, const B &b, const SA &sa, const SB &sb, int chunk_idx, bool init) {
    mma_k96_chunk<transpose::N, transpose::T, 2, D, A, B, SA, SB>(d, a, b, sa, sb, chunk_idx, init);
}

// ---------------------------------------------------------------------------
// Single-chunk NVFP4 K64 microscaling MMA (E8M0 / block32 scales).
//
// Mirrors `mma_k96_chunk` but issues exactly one K64 tcgen05 MMA for chunk
// `chunk_idx` of the (full, contiguous) A/B shared tiles. The K64 block32 scale
// addressing matches the core `mma` loop: two consecutive chunks share one
// scale panel (the panel index advances every 2 chunks) and alternate the
// scale-factor-id between 0 and 2. `init` selects acc=0 (initialize) vs acc=1
// (accumulate). No commit is performed; call `mma_k96_commit` after the final
// chunk (it is K-agnostic and just commits to the semaphore).
template<int trans_a, int n_trans_b, int ncta, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_k64_chunk(D &d, const A &a, const B &b, const SA &sa, const SB &sb, int chunk_idx, bool init) {
    using T_AB  = typename A::T;
    using T_SAB = typename SA::T;
    using T_D   = typename D::T;
    static_assert(std::is_same_v<T_AB, fp4e2m1_2>, "mma_k64_chunk is only for NVFP4 operands.");
    static_assert(std::is_same_v<T_SAB, fp8e8m0>, "mma_k64_chunk expects E8M0 (block32) scale factors.");
    constexpr int block_size = 32;
    constexpr int trans_b = 1 - n_trans_b;
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    constexpr int M_offset = M / 32 / ncta;
    constexpr int N_offset = N / 32;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    constexpr uint32_t idesc0 = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0, 64>();
    constexpr uint32_t idesc2 = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 2, 64>();
    const uint64_t ad = a_desc.template chunk_descriptor<64>(chunk_idx);
    const uint64_t bd = b_desc.template chunk_descriptor<64>(chunk_idx);
    const uint32_t sa_addr = sa.addr + (chunk_idx >> 1) * M_offset;
    const uint32_t sb_addr = sb.addr + (chunk_idx >> 1) * N_offset;
    const uint32_t idesc   = (chunk_idx & 1) ? idesc2 : idesc0;
    if (init)
        detail::tcgen05::template st_st<T_AB, T_SAB, 0, ncta, block_size>(
            d.addr, ad, bd, sa_addr, sb_addr, idesc);
    else
        detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
            d.addr, ad, bd, sa_addr, sb_addr, idesc);
}
// ABt, cta_group::2 convenience wrapper matching mma2_ABt_k96_chunk.
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k64_chunk(D &d, const A &a, const B &b, const SA &sa, const SB &sb, int chunk_idx, bool init) {
    mma_k64_chunk<transpose::N, transpose::T, 2, D, A, B, SA, SB>(d, a, b, sa, sb, chunk_idx, init);
}

} // namespace kittens
