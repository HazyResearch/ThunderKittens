/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in tensor memory.
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

namespace detail {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor
template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
struct instruction_descriptor {
    uint32_t desc = 0;
    if constexpr (sizeof(AB) == 2) { // kind::f16
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
        } else {
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
    }
    else {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
    }
    return desc;
};

template<int acc>
__device__ static inline void tmem_st(uint32_t d_tmem_addr, uint32_t a_tmem_addr, uint64_t b_desc, uint32_t idesc) {
    asm volatile(
        ".reg .pred p;\n" \
        "setp.eq.u32 p, 1, %4;\n" \
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
    ::  "r"(d_tmem_addr), "r"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
    );
}

template<int acc>
__device__ static inline void st_st(uint32_t d_tmem_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    asm volatile(
        ".reg .pred p;\n" \
        "setp.eq.u32 p, 1, %4;\n" \
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
    ::  "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
    );
}

__device__ static inline void commit(kittens::semaphore &sem) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
    ::  "l"(&sem)
    );
}

}

template<typename T_AB> constexpr int reduction_dimension = sizeof(T_AB) == 2 ? 16 : sizeof(T_AB) == 4 ? 8 : 32; // haven't added fp4 yet.

// RS matmul equivalent
template<int trans_a, int trans_b, ducks::tmem::all D, ducks::tmem::all A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {

    // Do everything here.
    constexpr int M = trans_a ? A::width : A::height;
    static_assert(M == D::height && (M == 4 || M == 8)); // output register is correctly sized

    constexpr int N = trans_b ? B::height : B::width;
    static_assert(N == D::width); // output register is correctly sized

    constexpr int K = trans_a ? A::height : A::width;
    static_assert((trans_b ? B::width : B::height) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<typename A::T>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, half>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, bf16>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, half>),
        "Currently unsupported type combination for matrix multiply."
    );
    uint32_t idesc = detail::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    if(laneid() == 0) {
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

        detail::template tmem_st<acc>(
            d.addr,
            a.chunk_addr(0),
            b_desc.chunk_descriptor(0),
            idesc
        );
        #pragma unroll
        for(int i = 1; i < K/red_dim; i++) {
            detail::template tmem_st<1>(
                d.addr,
                a.chunk_addr(i),
                b_desc.chunk_descriptor(i),
                idesc
            );
        }
        detail::commit(sem);
    }
    __syncwarp();
}
// SS matmul equivalent
template<int trans_a, int trans_b, ducks::tmem::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {

    // Do everything here.
    constexpr int M = trans_a ? A::width : A::height;
    static_assert(M == D::height && (M == 4 || M == 8)); // output register is correctly sized

    constexpr int N = trans_b ? B::height : B::width;
    static_assert(N == D::width); // output register is correctly sized

    constexpr int K = trans_a ? A::height : A::width;
    static_assert((trans_b ? B::width : B::height) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<typename A::T>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, half>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, bf16>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, half>),
        "Currently unsupported type combination for matrix multiply."
    );
    uint32_t idesc = detail::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    if(laneid() == 0) {
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

        detail::template st_st<acc>(
            d.addr,
            a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0),
            idesc
        );
        #pragma unroll
        for(int i = 1; i < K/red_dim; i++) {
            detail::template st_st<1>(
                d.addr,
                a_desc.chunk_descriptor(i),
                b_desc.chunk_descriptor(i),
                idesc
            );
        }
        detail::commit(sem);
    }
    __syncwarp();
}
template<int trans_a, int trans_b, ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}

template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<0, 1, D, A, B, 1>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<0, 0, D, A, B, 1>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<1, 1, D, A, B, 1>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<1, 0, D, A, B, 1>(d, a, b);
}

template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<0, 1, D, A, B, 0>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<0, 0, D, A, B, 0>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<1, 1, D, A, B, 0>(d, a, b);
}
template<ducks::tmem::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<1, 0, D, A, B, 0>(d, a, b);
}


} // namespace kittens