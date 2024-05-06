/**
 * @file
 * @brief Warpgroup matrix-multiply accumulate operations. These ops are necessary to achieve full utilization on H100 GPUs.
 */


 /*
 ### OPTIONS:

 REG+SMEM -> REG
 - mma_AB   (accum) [DONE]
 - mm_AB    (reset) [DONE]
 - mma_ABt  (accum) [DONE]
 - mm_ABt   (reset) [DONE]
 
 SMEM+SMEM -> REG
 - mma_AB   (accum) [DONE]
 - mm_AB    (reset) [DONE]
 - mma_ABt  (accum) [DONE]
 - mm_ABt   (reset) [DONE]
 - mma_AtB  (accum) [DONE]
 - mm_AtB   (reset) [DONE]
 - mma_AtBt (accum) [DONE]
 - mm_AtBt  (reset) [DONE]
 
Note: mma is an alias for mma_AB and dot is an alias for mma_ABt
*/

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
template<int N_DIV_4, int K, int M, ducks::wgmma::transposed L_B, int accumulate=1>
__device__ static inline void mma_AB(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                               const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                               const st_bf<K, M, L_B>                         &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 0, 1>;
    #pragma unroll
    for(int n = 0; n < N_DIV_4; n++) {
        rt_fl<1, M, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, n);
        base::rt_st(
            d_ref,
            a.tiles[n][0],
            base::b_desc(b, 0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[n][k],
                base::b_desc(b, k),
                1
            );
        }
    }
}
template<int N_DIV_4, int K, int M, ducks::wgmma::normal L_B>
__device__ static inline void mm_AB(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                              const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                              const st_bf<K, M, L_B>                         &b) {
    mm_AB<N_DIV_4, K, M, L_B, 0>(d, a, b);
}

template<int K, int M, ducks::wgmma::normal L_A, ducks::wgmma::transposed L_B, int accumulate=1>
__device__ static inline void mma_AB(rt_fl<1, M, ducks::rt_layout::row> &d,
                               const st_bf<4, K, L_A>                   &a,
                               const st_bf<K, M, L_B>                   &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 0, 1>;
    base::st_st(
        d,
        base::a_desc(a, 0),
        base::b_desc(b, 0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            base::a_desc(a, k),
            base::b_desc(b, k),
            1
        );
    }
}
template<int K, int M, ducks::wgmma::normal L_A, ducks::wgmma::transposed L_B>
__device__ static inline void mm_AB(rt_fl<1, M, ducks::rt_layout::row> &d,
                              const st_bf<4, K, L_A>                   &a,
                              const st_bf<K, M, L_B>                   &b) {
    mma_AB<K, M, L_A, L_B, 0>(d, a, b);
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
template<int N_DIV_4, int K, int M, ducks::wgmma::normal L_B, int accumulate=1>
__device__ static inline void mma_ABt(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                                const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                                const st_bf<M, K, L_B>                         &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 0, 0>;
    #pragma unroll
    for(int n = 0; n < N_DIV_4; n++) {
        rt_fl<1, M, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, n);
        base::rt_st(
            d_ref,
            a.tiles[n][0],
            base::b_desc(b, 0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[n][k],
                base::b_desc(b, k),
                1
            );
        }
    }
}
template<int N_DIV_4, int K, int M, ducks::wgmma::normal L_B>
__device__ static inline void mm_ABt(rt_fl<N_DIV_4, M, ducks::rt_layout::row> &d,
                               const rt_bf<N_DIV_4, K, ducks::rt_layout::row> &a,
                               const st_bf<M, K, L_B>                         &b) {
    mma_ABt<N_DIV_4, K, M, L_B, 0>(d, a, b);
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
template<int K, int M, ducks::wgmma::normal L_A, ducks::wgmma::normal L_B, int accumulate=1>
__device__ static inline void mma_ABt(rt_fl<1, M, ducks::rt_layout::row> &d,
                                const st_bf<4, K, L_A>                   &a,
                                const st_bf<M, K, L_B>                   &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 0, 0>;
    base::st_st(
        d,
        base::a_desc(a, 0),
        base::b_desc(b, 0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            base::a_desc(a, k),
            base::b_desc(b, k),
            1
        );
    }
}
template<int K, int M, ducks::wgmma::normal L_A, ducks::wgmma::normal L_B>
__device__ static inline void mm_ABt(rt_fl<1, M, ducks::rt_layout::row> &d,
                               const st_bf<4, K, L_A>                   &a,
                               const st_bf<M, K, L_B>                   &b) {
    mma_ABt<K, M, L_A, L_B, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A transposed.
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
template<int K, int M, ducks::wgmma::transposed L_A, ducks::wgmma::transposed L_B, int accumulate=1>
__device__ static inline void mma_AtB(rt_fl<1, M, ducks::rt_layout::row> &d,
                                const st_bf<K, 4, L_A>                   &a,
                                const st_bf<K, M, L_B>                   &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 1, 1>;
    base::st_st(
        d,
        base::a_desc(a, 0),
        base::b_desc(b, 0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            base::a_desc(a, k),
            base::b_desc(b, k),
            1
        );
    }
}
template<int K, int M, ducks::wgmma::transposed L_A, ducks::wgmma::transposed L_B>
__device__ static inline void mm_AtB(rt_fl<1, M, ducks::rt_layout::row> &d,
                               const st_bf<K, 4, L_A>                   &a,
                               const st_bf<K, M, L_B>                   &b) {
    mma_AtB<K, M, L_A, L_B, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A and B transposed.
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
template<int K, int M, ducks::wgmma::transposed L_A, ducks::wgmma::normal L_B, int accumulate=1>
__device__ static inline void mma_AtBt(rt_fl<1, M, ducks::rt_layout::row> &d,
                                 const st_bf<K, 4, L_A>                   &a,
                                 const st_bf<M, K, L_B>                   &b) {
    KITTENS_CHECK_WARPGROUP
    using base = kittens::wgmma::base<M, 1, 0>;
    base::st_st(
        d,
        base::a_desc(a, 0),
        base::b_desc(b, 0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            base::a_desc(a, k),
            base::b_desc(b, k),
            1
        );
    }
}
template<int K, int M, ducks::wgmma::transposed L_A, ducks::wgmma::normal L_B>
__device__ static inline void mm_AtBt(rt_fl<1, M, ducks::rt_layout::row> &d,
                                const st_bf<K, 4, L_A>                   &a,
                                const st_bf<M, K, L_B>                   &b) {
    mma_AtBt<K, M, L_A, L_B, 0>(d, a, b);
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
__device__ static inline void mma_fence(rt_fl<height, width, ducks::rt_layout::row> &dst) {
    KITTENS_CHECK_WARPGROUP
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
 template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP
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
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}