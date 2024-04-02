/**
 * @file
 * @brief Warpgroup matrix-multiply accumulate operations. These ops are necessary to achieve full utilization on H100 GPUs.
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

#include "base/base.cuh"

namespace kittens {
/**
 * @namespace warpgroup
 * 
 * @brief A namespace for operations undertaken collaboratively by a warpgroup.
 *
 * Warpgroup functions must be called together by a group of 4 warps, which complete it together.
 * Note that if this is not respected, behavior is undefined.
*/
namespace warpgroup {

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix multiply-accumulate operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function multiplies a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int accumulate, int N_DIV_4, int K, int M, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                            const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                            const st_bf<K, M, L_B>           &b) {
    #pragma unroll
    for(int n = 0; n < N_DIV_4; n++) {
        rt_fl<1, M, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, n);
        wgmma_base<M, 1>::rt_st(
            d_ref,
            a.tiles[n][0],
            b.descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            wgmma_base<M, 1>::rt_st(
                d_ref,
                a.tiles[n][k],
                b.descriptor(k),
                1
            );
        }
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
 * @param d[out] The destination register tile where the result is accumulated.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_accum(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
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
 * @param d[out] The destination register tile where the result is written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_reset(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<0, N_DIV_4, K, M, L_B>(d, a, b);
}
// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply-accumulate operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function multiplies a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int accumulate, int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma(rt_fl<1, M, ducks::rt_layout::row> &d,
                            const st_bf<4, K, L_A>           &a,
                            const st_bf<K, M, L_B>           &b) {
    wgmma_base<M, 1>::st_st(
        d,
        a.descriptor(0),
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M, 1>::st_st(
            d,
            a.descriptor(k),
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
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_accum(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<1, K, M, L_A, L_B>(d, a, b);
}
/**
 * @brief Perform matrix multiply-accumulate operation with reset behavior.
 *
 * This function is a wrapper around `mma` with the `accumulate` parameter set to 0, indicating that the result should overwrite `d`.
 *
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_reset(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<0, K, M, L_A, L_B>(d, a, b);
}

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int accumulate, int N_DIV_4, int K, int M, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                            const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                            const st_bf<M, K, L_B>           &b) {
    #pragma unroll
    for(int n = 0; n < N_DIV_4; n++) {
        rt_fl<1, M, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, n);
        wgmma_base<M, 0>::rt_st(
            d_ref,
            a.tiles[n][0],
            b.descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            wgmma_base<M, 0>::rt_st(
                d_ref,
                a.tiles[n][k],
                b.descriptor(k),
                1
            );
        }
    }
}
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function is a wrapper around `dot` with the `accumulate` parameter set to 1, indicating that the result should be accumulated into `d`.
 *
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_accum(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, N_DIV_4, K, M, L_B>(d, a, b);
}
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function is a wrapper around `dot` with the `accumulate` parameter set to 0, indicating that the result should overwrite `d`.
 *
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_reset(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, N_DIV_4, K, M, L_B>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int accumulate, int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot(rt_fl<1, M, ducks::rt_layout::row> &d,
                            const st_bf<4, K, L_A>           &a,
                            const st_bf<M, K, L_B>           &b) {
    wgmma_base<M, 0>::st_st(
        d,
        a.descriptor(0),
        b.descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        wgmma_base<M, 0>::st_st(
            d,
            a.descriptor(k),
            b.descriptor(k),
            1
        );
    }
}
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function is a wrapper around `dot` with the `accumulate` parameter set to 1, indicating that the result should be accumulated into `d`.
 *
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_accum(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, K, M, L_A, L_B>(d, a, b);
}
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function is a wrapper around `dot` with the `accumulate` parameter set to 0, indicating that the result should overwrite `d`.
 *
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_reset(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, K, M, L_A, L_B>(d, a, b);
}

/**
 * @brief Synchronize the warp group and ensure that all writes to shared memory are visible to all threads in the warp group.
 *
 * This function acts as a fence for shared memory operations, ensuring that all previous writes are visible before proceeding.
 * This function should be called before running wgmma::mma or wgmma::dot instructions.
 *
 * @tparam height The height of the matrix `dst`.
 * @tparam width The width of the matrix `dst`.
 * @param dst[in,out] The destination register-tile matrix to be synchronized.
 */
template<int height, int width>
__device__ inline void fence(rt_fl<height, width, ducks::rt_layout::row> &dst) {
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
 * @brief Commit the current set of warp group matrix multiply accumulate calls.
 */
__device__ inline void mma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Wait for the warp group to reach a synchronization point.
 *
 * This function stalls the current warpgroup until enough WGMMA committed groups have been completed.
 *
 * @tparam N The number of remaining active WGMMA committed groups allowed. This will stall until the number of active groups is less than or equal to N. Defaults to 0.
 */
template<int N=0>
__device__ inline void mma_async_wait() {
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}

}
}