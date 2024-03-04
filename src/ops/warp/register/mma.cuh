#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1))

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}

__device__ static inline void mma_base(rt_base<float2, rt_row_layout> &d,
                                 const rt_base<bf16_2, rt_row_layout> &a,
                                 const rt_base<bf16_2, rt_col_layout> &b, // in col-major mode
                                 const rt_base<float2, rt_row_layout> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
__device__ static inline void dot_base(rt_base<float2, rt_row_layout> &d,
                                 const rt_base<bf16_2, rt_row_layout> &a,
                                 const rt_base<bf16_2, rt_row_layout> &b, // in row-major mode
                                 const rt_base<float2, rt_row_layout> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}

template<int N, int K, int M>
__device__ static inline void mma(rt_fl<N, M, rt_row_layout> &d,
                            const rt_bf<N, K, rt_row_layout> &a,
                            const rt_bf<K, M, rt_col_layout> &b,
                            const rt_fl<N, M, rt_row_layout> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            mma_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                mma_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<int N, int K, int M>
__device__ static inline void dot(rt_fl<N, M, rt_row_layout> &d,
                            const rt_bf<N, K, rt_row_layout> &a,
                            const rt_bf<M, K, rt_row_layout> &b, // notice row and (M, K) instead of col and (K, M)
                            const rt_fl<N, M, rt_row_layout> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            dot_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                dot_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

}

/*

WMMA call for reference

asm volatile(
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-and-accumulate-instruction-wmma-mma
    "wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32 " \
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }, " \
    "{%8,  %9,  %10, %11}, " \
    "{%12, %13, %14, %15}, " \
    "{%16, %17, %18, %19, %20, %21, %22, %23};"

    // D matrix
:   "+f"(d.data[0].x), "+f"(d.data[0].y),
    "+f"(d.data[1].x), "+f"(d.data[1].y),
    "+f"(d.data[2].x), "+f"(d.data[2].y),
    "+f"(d.data[3].x), "+f"(d.data[3].y)

    // A matrix
:   "r"(*(uint32_t*)(&a.data[0])), "r"(*(uint32_t*)(&a.data[1])),
    "r"(*(uint32_t*)(&a.data[2])), "r"(*(uint32_t*)(&a.data[3])),

    // B matrix
    "r"(*(uint32_t*)(&b.data[0])), "r"(*(uint32_t*)(&b.data[1])),
    "r"(*(uint32_t*)(&b.data[2])), "r"(*(uint32_t*)(&b.data[3])),

    // C matrix
    "f"(c.data[0].x), "f"(c.data[0].y),
    "f"(c.data[1].x), "f"(c.data[1].y),
    "f"(c.data[2].x), "f"(c.data[2].y),
    "f"(c.data[3].x), "f"(c.data[3].y)
);

*/