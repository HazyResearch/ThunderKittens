#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

#include "base/base.cuh"

namespace kittens {
namespace warpgroup {

/**
 * @brief Perform matrix multiply-accumulate operation using warp group matrix multiply-accumulate (WGMA) primitives.
 *
 * This function multiplies a register-tile matrix `a` with a shared-tile matrix `b` and accumulates the result into a register-tile matrix `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d The destination register-tile matrix where the result is accumulated or written.
 * @param a The source register-tile matrix to be multiplied.
 * @param b The source shared-tile matrix to be multiplied.
 */
template<int accumulate, int N_DIV_4, int K, int M, st_wgmma_col_layout L_B>
__device__ static inline void mma(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                            const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                            const st_bf<K, M, L_B>           &b) {
    wgmma_base<M>::rt_st(
        d,
        a.tiles[0][0],
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M>::rt_st(
            d,
            a.tiles[0][k],
            b.descriptor(k),
            1
        );
    }
}

/**
 * @brief Perform matrix multiply-accumulate operation with accumulation behavior.
 *
 * This function is a wrapper around `mma` with the `accumulate` parameter set to 1, indicating that the result should be accumulated into `d`.
 *
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d The destination register-tile matrix where the result is accumulated.
 * @param a The source register-tile matrix to be multiplied.
 * @param b The source shared-tile matrix to be multiplied.
 */
template<int N_DIV_4, int K, int M, st_wgmma_col_layout L_B>
__device__ static inline void mma_accum(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<1, N_DIV_4, K, M, L_B>(d, a, b);
}

/**
 * @brief Perform matrix multiply-accumulate operation with reset behavior.
 *
 * This function is a wrapper around `mma` with the `accumulate` parameter set to 0, indicating that the result should overwrite `d`.
 *
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d The destination register-tile matrix where the result is written.
 * @param a The source register-tile matrix to be multiplied.
 * @param b The source shared-tile matrix to be multiplied.
 */
template<int N_DIV_4, int K, int M, st_wgmma_col_layout L_B>
__device__ static inline void mma_reset(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<0, N_DIV_4, K, M, L_B>(d, a, b);
}

// [(shared, shared) -> register] edition
template<int accumulate, int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_col_layout L_B>
__device__ static inline void mma(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                            const st_bf<4*N_DIV_4, K, L_A>           &a,
                            const st_bf<K, M, L_B>           &b) {
    wgmma_base<M>::st_st(
        d,
        a.descriptor(0),
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M>::st_st(
            d,
            a.descriptor(k),
            b.descriptor(k),
            1
        );
    }
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_col_layout L_B>
__device__ static inline void mma_accum(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const st_bf<4*N_DIV_4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<1, N_DIV_4, K, M, L_A, L_B>(d, a, b);
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_col_layout L_B>
__device__ static inline void mma_reset(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const st_bf<4*N_DIV_4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<0, N_DIV_4, K, M, L_A, L_B>(d, a, b);
}

// [(register, shared) -> register] edition
template<int accumulate, int N_DIV_4, int K, int M, st_wgmma_row_layout L_B>
__device__ static inline void dot(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                            const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                            const st_bf<M, K, L_B>           &b) {
    wgmma_base<M>::rt_st(
        d,
        a.tiles[0][0],
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M>::rt_st(
            d,
            a.tiles[0][k],
            b.descriptor(k),
            1
        );
    }
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_B>
__device__ static inline void dot_accum(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, N_DIV_4, K, M, L_B>(d, a, b);
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_B>
__device__ static inline void dot_reset(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, N_DIV_4, K, M, L_B>(d, a, b);
}
// [(shared, shared) -> register] edition
template<int accumulate, int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_row_layout L_B>
__device__ static inline void dot(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                            const st_bf<4*N_DIV_4, K, L_A>           &a,
                            const st_bf<M, K, L_B>           &b) {
    wgmma_base<M>::st_st(
        d,
        a.descriptor(0),
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M>::st_st(
            d,
            a.descriptor(k),
            b.descriptor(k),
            1
        );
    }
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_row_layout L_B>
__device__ static inline void dot_accum(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const st_bf<4*N_DIV_4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, N_DIV_4, K, M, L_A, L_B>(d, a, b);
}
template<int N_DIV_4, int K, int M, st_wgmma_row_layout L_A, st_wgmma_row_layout L_B>
__device__ static inline void dot_reset(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const st_bf<4*N_DIV_4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, N_DIV_4, K, M, L_A, L_B>(d, a, b);
}

/**
 * @brief Synchronize the warp group and ensure that all writes to shared memory are visible to all threads in the warp group.
 *
 * This function acts as a fence for shared memory operations, ensuring that all previous writes are visible before proceeding.
 *
 * @tparam height The height of the matrix `dst`.
 * @tparam width The width of the matrix `dst`.
 * @param dst The destination register-tile matrix to be synchronized.
 */
template<int height, int width>
__device__ inline void fence(rt_fl<height, width, rt_row_layout> &dst) {
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory"); // still not really sure if we need this
    #pragma unroll
    for(int i = 0; i < height; i++) {
        #pragma unroll
        for(int j = 0; j < width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                asm volatile("" : "+f"(dst.tiles[i][j].data[k].x) :: "memory");
                asm volatile("" : "+f"(dst.tiles[i][j].data[k].y) :: "memory");
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Commit the current warp group to a synchronization point.
 *
 * This function ensures that all threads in the warp group reach a synchronization point before any thread can proceed.
 */
__device__ inline void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Wait for the warp group to reach a synchronization point.
 *
 * This function stalls the current thread until all threads in the warp group have reached the specified synchronization point.
 *
 * @tparam N The synchronization point to wait for.
 */
template<int N=0>
__device__ inline void mma_async_wait() {
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}

}
}
