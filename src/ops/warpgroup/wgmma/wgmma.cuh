#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

#include "base/base.cuh"

namespace kittens {
namespace warpgroup {

// [(register, shared) -> register] edition
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
template<int N_DIV_4, int K, int M, st_wgmma_col_layout L_B>
__device__ static inline void mma_accum(rt_fl<N_DIV_4, M, rt_row_layout> &d,
                                  const rt_bf<N_DIV_4, K, rt_row_layout> &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<1, N_DIV_4, K, M, L_B>(d, a, b);
}
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

__device__ inline void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N=0>
__device__ inline void mma_async_wait() {
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}

}
}