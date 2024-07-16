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
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::wgmma::st_transposed B, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 0, 1>;
    kittens::wgmma::descriptor<B, 1> b_desc(b);

    // Do it
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, 1, N, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::wgmma::st_transposed B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::wgmma::st_normal A, ducks::wgmma::st_transposed B, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 0, 1>;
    kittens::wgmma::descriptor<A, 0> a_desc(a);
    kittens::wgmma::descriptor<B, 1> b_desc(b);

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
}
template<ducks::rt::row_layout D, ducks::wgmma::st_normal A, ducks::wgmma::st_transposed B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 0>(d, a, b);
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
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::wgmma::st_normal B, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 0, 0>;
    kittens::wgmma::descriptor<B, 0> b_desc(b);

    // Do it
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, 1, N, ducks::rt_layout::row> &d_ref = subtile_inplace<1>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::wgmma::st_normal B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 0>(d, a, b);
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
template<ducks::rt::row_layout D, ducks::wgmma::st_normal A, ducks::wgmma::st_normal B, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 0, 0>;
    kittens::wgmma::descriptor<A, 0> a_desc(a);
    kittens::wgmma::descriptor<B, 0> b_desc(b);

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
}
template<ducks::rt::row_layout D, ducks::wgmma::st_normal A, ducks::wgmma::st_normal B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 0>(d, a, b);
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
template<ducks::rt::row_layout D, ducks::wgmma::st_transposed A, ducks::wgmma::st_transposed B, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::height;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 1, 1>;
    kittens::wgmma::descriptor<A, 1> a_desc(a);
    kittens::wgmma::descriptor<B, 1> b_desc(b);

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
}
template<ducks::rt::row_layout D, ducks::wgmma::st_transposed A, ducks::wgmma::st_transposed B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A and B transposed.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam D The destination register tile type.
 * @tparam A The source shared tile type.
 * @tparam B The source shared tile type.
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 */
template<ducks::rt::row_layout D, ducks::wgmma::st_transposed A, ducks::wgmma::st_normal B, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::height;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::wgmma::base<T_D, T_AB, N, 1, 0>;
    kittens::wgmma::descriptor<A, 1> a_desc(a);
    kittens::wgmma::descriptor<B, 0> b_desc(b);

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
}
template<ducks::rt::row_layout D, ducks::wgmma::st_transposed A, ducks::wgmma::st_normal B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    mma_AtBt<D, A, B, 0>(d, a, b);
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
template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                if constexpr(std::is_same_v<typename D::T, float>) {
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].y) :: "memory");
                } else {
                    asm volatile("" : "+r"(*(uint32_t*)&dst.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP
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