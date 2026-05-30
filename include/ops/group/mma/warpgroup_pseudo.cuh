/**
 * @file
 * @brief Warpgroup-shaped MMA fallback for targets without WGMMA.
 *
 * This file is included inside kittens::group<N>. It provides a small subset of
 * the warpgroup shared/shared -> register MMA API on architectures where the
 * hardware WGMMA instruction is unavailable. The implementation keeps the same
 * warpgroup tiling shape, but each warp loads 16x16 shared-memory fragments and
 * executes warp-scope HMMA instructions.
 */

#pragma once

// The pseudo implementation is synchronous. These functions intentionally keep
// the WGMMA API surface available for source compatibility.
template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    asm volatile("" : : "l"(&dst) : "memory");
}
template<ducks::crt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    asm volatile("" : : "l"(&dst) : "memory");
}
template<typename T=kittens::ducks::default_type>
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP
    asm volatile("" ::: "memory");
}
template<typename T=kittens::ducks::default_type>
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP
    asm volatile("" ::: "memory");
}
template<int N=0>
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
    asm volatile("" ::: "memory");
}

template<typename T>
static constexpr bool pseudo_wgmma_16bit_float_v = std::is_same_v<T, bf16> || std::is_same_v<T, half>;

template<typename T>
static constexpr bool pseudo_wgmma_8bit_int_v = std::is_same_v<T, int8> || std::is_same_v<T, uint8>;

template<typename T>
static constexpr bool pseudo_wgmma_supported_operand_v = pseudo_wgmma_16bit_float_v<T> || pseudo_wgmma_8bit_int_v<T>;

template<typename T_D, typename T_AB>
static constexpr bool pseudo_wgmma_supported_accum_v =
    (std::is_same_v<T_D, float> && pseudo_wgmma_16bit_float_v<T_AB>) ||
    (std::is_same_v<T_D, half> && std::is_same_v<T_AB, half>) ||
    (std::is_same_v<T_D, int> && pseudo_wgmma_8bit_int_v<T_AB>);

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ static inline void pseudo_wgmma_load_16x16(RT &dst, const ST &src, int row_offset, int col_offset) {
    KITTENS_CHECK_WARPGROUP
    static_assert(RT::rows == kittens::TILE_ROW_DIM<typename RT::T>);
    static_assert(RT::cols == kittens::TILE_COL_DIM<typename RT::T>);
    static_assert(std::is_same_v<typename RT::T, typename ST::T>);
    static_assert(pseudo_wgmma_supported_operand_v<typename RT::T>,
                  "Pseudo warpgroup MMA currently supports bf16/half/int8/uint8 operands only.");

    using T2 = typename RT::dtype;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<U *>(&src.data[0]))
    );
    int warp_laneid = ::kittens::laneid();
    U2 tmp[4];

    int row = row_offset + (warp_laneid % 16);
    int col = col_offset + (warp_laneid / 16) * (kittens::TILE_COL_DIM<U> / 2);

    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));

    dst.tiles[0][0].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
    dst.tiles[0][0].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
    dst.tiles[0][0].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
    dst.tiles[0][0].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
}

template<ducks::rt::col_layout RT, ducks::st::all ST>
__device__ static inline void pseudo_wgmma_load_16x16(RT &dst, const ST &src, int row_offset, int col_offset) {
    KITTENS_CHECK_WARPGROUP
    static_assert(RT::rows == kittens::TILE_ROW_DIM<typename RT::T>);
    static_assert(RT::cols == kittens::TILE_COL_DIM<typename RT::T>);
    static_assert(std::is_same_v<typename RT::T, typename ST::T>);
    static_assert(pseudo_wgmma_16bit_float_v<typename RT::T>,
                  "Pseudo warpgroup MMA supports col-layout shared fragment loads for bf16/half only. Integer MMA uses ABt row-layout operands.");

    using T2 = typename RT::dtype;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;

    uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<U *>(&src.data[0]))
    );
    int warp_laneid = ::kittens::laneid();
    U2 tmp[4];

    int row = row_offset + (warp_laneid % 16);
    int col = col_offset + (warp_laneid / 16) * 8;

    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));

    dst.tiles[0][0].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
    dst.tiles[0][0].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
    dst.tiles[0][0].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
    dst.tiles[0][0].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
}

template<ducks::rt_base::all D, ducks::rt_base::all A, ducks::rt_base::all B>
__device__ static inline void pseudo_wgmma_mma_AB_base(D &d, const A &a, const B &b) {
    group<1>::mma_AB_base(d, a, b, d);
}

template<ducks::rt_base::all D, ducks::rt_base::all A, ducks::rt_base::all B>
__device__ static inline void pseudo_wgmma_mma_ABt_base(D &d, const A &a, const B &b) {
    if constexpr (std::is_same_v<typename D::T, half>) {
        static_assert(std::is_same_v<typename A::T, half> && std::is_same_v<typename B::T, half>);
        hmma16816(
            d.data[0], d.data[1],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[0], b.data[2],
            d.data[0], d.data[1]
        );
        hmma16816(
            d.data[2], d.data[3],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[1], b.data[3],
            d.data[2], d.data[3]
        );
    }
    else {
        group<1>::mma_ABt_base(d, a, b, d);
    }
}

template<ducks::rt_base::all D, ducks::rt_base::all A, ducks::rt_base::all B>
__device__ static inline void pseudo_wgmma_mma_AtB_base(D &d, const A &a, const B &b) {
    if constexpr (std::is_same_v<typename D::T, half>) {
        static_assert(std::is_same_v<typename A::T, half> && std::is_same_v<typename B::T, half>);
        hmma16816(
            d.data[0], d.data[1],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[0], b.data[2],
            d.data[0], d.data[1]
        );
        hmma16816(
            d.data[2], d.data[3],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[1], b.data[3],
            d.data[2], d.data[3]
        );
    }
    else {
        group<1>::mma_AtB_base(d, a, b, d);
    }
}

template<ducks::rt_base::all D, ducks::rt_base::all A, ducks::rt_base::all B>
__device__ static inline void pseudo_wgmma_mma_AtBt_base(D &d, const A &a, const B &b) {
    if constexpr (std::is_same_v<typename D::T, half>) {
        static_assert(std::is_same_v<typename A::T, half> && std::is_same_v<typename B::T, half>);
        hmma16816(
            d.data[0], d.data[1],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[0], b.data[2],
            d.data[0], d.data[1]
        );
        hmma16816(
            d.data[2], d.data[3],
            a.data[0], a.data[1], a.data[2], a.data[3],
            b.data[1], b.data[3],
            d.data[2], d.data[3]
        );
    }
    else {
        group<1>::mma_AtBt_base(d, a, b, d);
    }
}

template<int trans_A, int trans_B, ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int accumulate>
__device__ static inline void pseudo_wgmma_st_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP

    static_assert(pseudo_wgmma_supported_accum_v<typename D::T, typename A::T>,
                  "Pseudo warpgroup MMA supports bf16/half operands with fp32 accumulators, half operands with half accumulators, or int8/uint8 operands with int accumulators.");
    static_assert(pseudo_wgmma_supported_operand_v<typename A::T>, "Pseudo warpgroup MMA currently supports bf16/half/int8/uint8 A operands only.");
    static_assert(std::is_same_v<typename A::T, typename B::T>, "Pseudo warpgroup MMA operands must have matching dtype.");
    static_assert(!pseudo_wgmma_8bit_int_v<typename A::T> || (trans_A == transpose::N && trans_B == transpose::T),
                  "Pseudo integer warpgroup MMA currently supports ABt only.");

    constexpr int BM = GROUP_WARPS * D::rows;
    constexpr int BN = D::cols;
    constexpr int BK = trans_A == transpose::N ? A::cols : A::rows;

    if constexpr (trans_A == transpose::N) {
        static_assert(A::rows == BM, "A rows must equal GROUP_WARPS * D rows.");
    }
    else {
        static_assert(A::cols == BM, "Transposed A columns must equal GROUP_WARPS * D rows.");
    }
    if constexpr (trans_B == transpose::N) {
        static_assert(B::rows == BK, "B rows must match the reduction dimension.");
        static_assert(B::cols == BN, "B columns must match D columns.");
    }
    else {
        static_assert(B::cols == BK, "Transposed B columns must match the reduction dimension.");
        static_assert(B::rows == BN, "Transposed B rows must match D columns.");
    }

    static_assert(D::rows % kittens::TILE_ROW_DIM<typename D::T> == 0, "D rows must be divisible by the base tile row dimension.");
    static_assert(D::cols % kittens::TILE_COL_DIM<typename D::T> == 0, "D columns must be divisible by the base tile column dimension.");
    static_assert(BK % kittens::TILE_COL_DIM<typename A::T> == 0, "Reduction dimension must be divisible by the operand base tile column dimension.");

    constexpr int M_FRAGS = D::height;
    constexpr int N_FRAGS = D::width;
    constexpr int K_FRAGS = BK / kittens::TILE_COL_DIM<typename A::T>;
    using a_layout = std::conditional_t<trans_A == transpose::N, ducks::rt_layout::row, ducks::rt_layout::col>;
    using b_layout = std::conditional_t<trans_B == transpose::N, ducks::rt_layout::col, ducks::rt_layout::row>;
    using operand_t = typename A::T;

    if constexpr (!accumulate) {
        group<1>::zero(d);
    }

    const int warp_m_offset = warpid() * D::rows;

    #pragma unroll
    for(int mi = 0; mi < M_FRAGS; mi++) {
        #pragma unroll
        for(int kk = 0; kk < K_FRAGS; kk++) {
            rt<operand_t, kittens::TILE_ROW_DIM<operand_t>, kittens::TILE_COL_DIM<operand_t>, a_layout> a_frag;
            pseudo_wgmma_load_16x16(
                a_frag,
                a,
                trans_A == transpose::N ? warp_m_offset + mi * kittens::TILE_ROW_DIM<operand_t> : kk * kittens::TILE_ROW_DIM<operand_t>,
                trans_A == transpose::N ? kk * kittens::TILE_COL_DIM<operand_t> : warp_m_offset + mi * kittens::TILE_COL_DIM<operand_t>
            );

            #pragma unroll
            for(int nj = 0; nj < N_FRAGS; nj++) {
                rt<operand_t, kittens::TILE_ROW_DIM<operand_t>, kittens::TILE_COL_DIM<operand_t>, b_layout> b_frag;
                pseudo_wgmma_load_16x16(
                    b_frag,
                    b,
                    trans_B == transpose::N ? kk * kittens::TILE_ROW_DIM<operand_t> : nj * kittens::TILE_ROW_DIM<operand_t>,
                    trans_B == transpose::N ? nj * kittens::TILE_COL_DIM<operand_t> : kk * kittens::TILE_COL_DIM<operand_t>
                );

                if constexpr (trans_A == transpose::N && trans_B == transpose::N) {
                    pseudo_wgmma_mma_AB_base(d.tiles[mi][nj], a_frag.tiles[0][0], b_frag.tiles[0][0]);
                }
                else if constexpr (trans_A == transpose::N && trans_B == transpose::T) {
                    pseudo_wgmma_mma_ABt_base(d.tiles[mi][nj], a_frag.tiles[0][0], b_frag.tiles[0][0]);
                }
                else if constexpr (trans_A == transpose::T && trans_B == transpose::N) {
                    pseudo_wgmma_mma_AtB_base(d.tiles[mi][nj], a_frag.tiles[0][0], b_frag.tiles[0][0]);
                }
                else {
                    pseudo_wgmma_mma_AtBt_base(d.tiles[mi][nj], a_frag.tiles[0][0], b_frag.tiles[0][0]);
                }
            }
        }
    }
}

template<int trans_B, ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int accumulate>
__device__ static inline void pseudo_wgmma_rt_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP

    static_assert(pseudo_wgmma_supported_accum_v<typename D::T, typename A::T>,
                  "Pseudo warpgroup MMA supports bf16/half operands with fp32 accumulators, half operands with half accumulators, or int8/uint8 operands with int accumulators.");
    static_assert(pseudo_wgmma_supported_operand_v<typename A::T>, "Pseudo warpgroup MMA currently supports bf16/half/int8/uint8 A operands only.");
    static_assert(std::is_same_v<typename A::T, typename B::T>, "Pseudo warpgroup MMA operands must have matching dtype.");
    static_assert(!pseudo_wgmma_8bit_int_v<typename A::T> || trans_B == transpose::T,
                  "Pseudo integer warpgroup register/shared MMA currently supports ABt only.");
    static_assert(D::rows == A::rows, "D rows must match A rows.");
    static_assert(D::cols == (trans_B == transpose::N ? B::cols : B::rows), "D columns must match B's logical N dimension.");
    static_assert(A::cols == (trans_B == transpose::N ? B::rows : B::cols), "A columns must match B's logical K dimension.");

    constexpr int K_FRAGS = A::width;
    constexpr int N_FRAGS = D::width;
    using b_layout = std::conditional_t<trans_B == transpose::N, ducks::rt_layout::col, ducks::rt_layout::row>;
    using operand_t = typename A::T;

    if constexpr (!accumulate) {
        group<1>::zero(d);
    }

    #pragma unroll
    for(int mi = 0; mi < D::height; mi++) {
        #pragma unroll
        for(int kk = 0; kk < K_FRAGS; kk++) {
            #pragma unroll
            for(int nj = 0; nj < N_FRAGS; nj++) {
                rt<operand_t, kittens::TILE_ROW_DIM<operand_t>, kittens::TILE_COL_DIM<operand_t>, b_layout> b_frag;
                pseudo_wgmma_load_16x16(
                    b_frag,
                    b,
                    trans_B == transpose::N ? kk * kittens::TILE_ROW_DIM<operand_t> : nj * kittens::TILE_ROW_DIM<operand_t>,
                    trans_B == transpose::N ? nj * kittens::TILE_COL_DIM<operand_t> : kk * kittens::TILE_COL_DIM<operand_t>
                );

                if constexpr (trans_B == transpose::N) {
                    pseudo_wgmma_mma_AB_base(d.tiles[mi][nj], a.tiles[mi][kk], b_frag.tiles[0][0]);
                }
                else {
                    pseudo_wgmma_mma_ABt_base(d.tiles[mi][nj], a.tiles[mi][kk], b_frag.tiles[0][0]);
                }
            }
        }
    }
}

/**
 * @brief Pseudo warpgroup matrix multiply-accumulate: D += A @ B.
 *
 * Shape contract:
 * - A is the warpgroup-level shared tile with shape BM x BK.
 * - B is the warpgroup-level shared tile with shape BK x BN.
 * - D is the current warp's accumulator with shape WM x BN.
 * - BM == GROUP_WARPS * WM, and GROUP_WARPS must be 4.
 *
 * Internally, A and B are loaded as 16x16 fragments on demand; only D is kept
 * as a full register tile.
 */
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_st_st<transpose::N, transpose::N, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}

/**
 * @brief Pseudo warpgroup matrix multiply: D = A @ B.
 */
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_rt_st<transpose::N, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_st_st<transpose::N, transpose::T, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_rt_st<transpose::T, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_st_st<transpose::T, transpose::N, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    if constexpr (fence) { mma_fence(d); }
    pseudo_wgmma_st_st<transpose::T, transpose::T, D, A, B, accumulate>(d, a, b);
    mma_commit_group();
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}

template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b);
        } else {
            mma_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b);
        } else {
            mma_AB(d, a, b);
        }
    }
}
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mm_AtBt(d, a, b);
        } else {
            mm_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mm_ABt(d, a, b);
        } else {
            mm_AB(d, a, b);
        }
    }
}
