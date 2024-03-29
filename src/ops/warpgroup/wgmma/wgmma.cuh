#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

#include "base/base.cuh"

namespace kittens {
namespace warpgroup {

// [(register, shared) -> register] edition
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
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_accum(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<K, M, L_B>                         &b) {
    mma<1, N_DIV_4, K, M, L_B>(d, a, b);
}
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_reset(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<K, M, L_B>                         &b) {
    mma<0, N_DIV_4, K, M, L_B>(d, a, b);
}
// [(shared, shared) -> register] edition
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
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_accum(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<1, K, M, L_A, L_B>(d, a, b);
}
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_col L_B>
__device__ static inline void mma_reset(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<K, M, L_B>           &b) {
    mma<0, K, M, L_A, L_B>(d, a, b);
}

// [(register, shared) -> register] edition
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
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_accum(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, N_DIV_4, K, M, L_B>(d, a, b);
}
template<int N_DIV_4, int K, int M, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_reset(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                  const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, N_DIV_4, K, M, L_B>(d, a, b);
}
// [(shared, shared) -> register] edition
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
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_accum(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<1, K, M, L_A, L_B>(d, a, b);
}
template<int K, int M, ducks::st_layout::wgmma_row L_A, ducks::st_layout::wgmma_row L_B>
__device__ static inline void dot_reset(rt_fl<1, M, ducks::rt_layout::row> &d,
                                  const st_bf<4, K, L_A>           &a,
                                  const st_bf<M, K, L_B>           &b) {
    dot<0, K, M, L_A, L_B>(d, a, b);
}

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

__device__ inline void mma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N=0>
__device__ inline void mma_async_wait() {
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}

}
}