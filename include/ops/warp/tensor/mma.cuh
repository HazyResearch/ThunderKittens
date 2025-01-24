/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in tensor memory.
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor
template<typename D, typename AB, int M, int K, int N, bool trans_a, bool trans_b, bool neg=false>
struct instruction_descriptor {
    uint32_t desc;
    __device__ inline instruction_descriptor() {
    desc = 0;
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

// RS matmul equivalent
template<int trans_a, int trans_b, ducks::tmem::all D, ducks::tmem::all A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {

    // Do everything here.
    if(laneid() == 0) {

    }

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
    using T_AB = A::T;
    using T_D  = D::T;
    static_assert(
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, half>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, bf16>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, half>),
        "Currently unsupported type combination for matrix multiply."
    );
    uint32_t idesc = instruction_descriptor<T_D, T_AB, M, K, N, trans_a, trans_b, false>().desc;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

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