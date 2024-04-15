/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Perform the HMMA.16816 operation.
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
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

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, rt_row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, rt_row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, rt_col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, rt_row_layout> accumulator matrix.
 */
__device__ static inline void mma_base(rt_base<float2, ducks::rt_layout::row> &d,
                                 const rt_base<bf16_2, ducks::rt_layout::row> &a,
                                 const rt_base<bf16_2, ducks::rt_layout::col> &b, // in col-major mode
                                 const rt_base<float2, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, rt_row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, rt_row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, rt_row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, rt_row_layout> accumulator matrix.
 */
__device__ static inline void dot_base(rt_base<float2, ducks::rt_layout::row> &d,
                                 const rt_base<bf16_2, ducks::rt_layout::row> &a,
                                 const rt_base<bf16_2, ducks::rt_layout::row> &b, // in row-major mode
                                 const rt_base<float2, ducks::rt_layout::row> &c) {
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

/**
 * @brief Matrix multiply-accumulate operation for row layout.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, rt_row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, rt_row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, rt_col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, rt_row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma(rt_fl<N, M, ducks::rt_layout::row> &d,
                            const rt_bf<N, K, ducks::rt_layout::row> &a,
                            const rt_bf<K, M, ducks::rt_layout::col> &b,
                            const rt_fl<N, M, ducks::rt_layout::row> &c) {
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
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, rt_row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, rt_row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, rt_row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, rt_row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void dot(rt_fl<N, M, ducks::rt_layout::row> &d,
                            const rt_bf<N, K, ducks::rt_layout::row> &a,
                            const rt_bf<M, K, ducks::rt_layout::row> &b, // notice row and (M, K) instead of col and (K, M)
                            const rt_fl<N, M, ducks::rt_layout::row> &c) {
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