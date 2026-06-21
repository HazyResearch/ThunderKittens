/**
 * @file
 * @brief SM120 (consumer Blackwell, e.g. GB10 / sm_121) shim for the warpgroup
 *        (group<4>) WGMMA API.
 *
 * SM120 has neither Hopper's `wgmma.mma_async` nor datacenter Blackwell's
 * `tcgen05` tensor-memory MMA. The native way to drive its tensor cores is the
 * warp-level `mma.sync` path in `warp.cuh`. This file re-exposes the
 * `warpgroup::mma_*` / `mm_*` surface that H100 kernels and the `prototype/`
 * LCF templates are written against, lowering each call to per-warp
 * `warp::mma_*` so those kernels compile and run unchanged.
 *
 * Key facts that make this a correctness-preserving lowering:
 *   - A 64-row warpgroup output tile is laid out as four 16-row bands, with
 *     warp `w` owning rows [16w, 16w+16). This is exactly how `group<4>::load`
 *     distributes a shared tile and how the WGMMA D fragment is partitioned, so
 *     a per-warp `warp::mma` over band `w` yields the same register layout WGMMA
 *     would have produced.
 *   - `group<1>::load` into a col-layout (or row-layout, for ABt) register tile
 *     performs the shared->register transpose-load via ldsm, matching the
 *     operand layouts `warp::mma_*` expects.
 *
 * Performance note: this is synchronous. WGMMA's async overlap (the basis of the
 * LCF producer/consumer pipeline) is lost, and shared operands are staged
 * through registers. Kernels are CORRECT but not pipeline-optimal; hot kernels
 * should be tuned per-kernel on top of this.
 *
 * RESERVED NAMED BARRIERS: mma_async_wait() uses bar.sync ids 12..15 (one per
 * warpgroup). Kernels that use this shim must keep their own named barriers in
 * 0..11. See the note on mma_async_wait below.
 *
 * Dtype coverage: bf16 only -- enforced by the static_assert below for every
 * variant, including AB. The SM120 warp-level mma.sync path lacks the
 * half->fp32 / fp8 / int8 base ops these warpgroup variants would need, so
 * those dtypes are intentionally absent here.
 */

// ---------------------------------------------------------------------------
//  Async-pipeline control ops -> no-ops / syncs (everything here is synchronous)
// ---------------------------------------------------------------------------

template<ducks::rt::all D> __device__ static inline void mma_fence(D &dst) {}
__device__ static inline void mma_fence() {}
__device__ static inline void mma_commit_group() {}
// On H100, mma_async_wait is wgmma.wait_group.sync.aligned -- a WARPGROUP-WIDE
// synchronization. The shim's MMA is synchronous per warp, but we still must
// barrier all 4 warps here: kernels signal shared-memory reuse (e.g. arrive() on
// a "compute done" semaphore from lane 0) right after this call, so without the
// barrier warps 1-3 may still be reading A/B from shared when the producer
// overwrites it (a WAR race).
//
// !!! RESERVED NAMED BARRIERS: this shim uses `bar.sync` id `8 + groupid()` (one
// per warpgroup). A named barrier is a block-global resource (ids 0..15) with no
// allocator, so any group that issues `bar.sync` on the same id with a different
// thread cohort can cross-release this barrier (deadlock or WAR-race corruption).
// Base 8 is chosen to be DISJOINT from the two barrier sets a shim-using kernel
// also touches:
//   - the LCF/LCSF/LCSC prototype templates (which the gemm runs on) use ids
//     13 (producers::sync), 14 (consumers::sync), 15 (everyone::sync);
//   - hand-written kernels use low ids -- mha_h100: groupid+4 (4/5/6) and a
//     group<8> barrier on id 10; gemm finish: groupid+4 (4..7).
// On GB10 only <=2 consumer warpgroups fit the 99 KB smem budget, so this uses
// ids 8,9 -- clean against all of the above. (An earlier version used 12+groupid
// = 12..15, which ALIASED the LCF template's 13/14/15: in the bf16_h100 gemm
// (LCF + this shim) consumer warpgroup 1's mma_async_wait collided with the
// producer's producers::sync(13). That is now fixed.) FORWARD-LOOKING CAVEAT:
// ids 8..11 brush mha's id 10, so a future kernel that combines this shim's mma
// with >2 consumer warpgroups AND a group<8>::sync(10) must re-audit. Kernels
// using this shim MUST keep their own named barriers disjoint from `8+groupid()`.
template<int N=0> __device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
    sync(8 + groupid());
}

// ---------------------------------------------------------------------------
//  Dtype guard: the SM120 warp-level mma.sync path has complete coverage only
//  for bf16 inputs. fp16/fp8/int8 warpgroup MMA is not implemented here (the
//  warp::mma_*_base overloads in warp.cuh are incomplete for those on SM120).
//  Without this check, an unsupported type produces a cryptic "no instance of
//  overloaded function warp::mma_*" error; this turns it into a clear message.
// ---------------------------------------------------------------------------
#define KITTENS_SM120_WG_MMA_DTYPE_CHECK(T) \
    static_assert(std::is_same_v<T, kittens::bf16>, \
        "SM120 (GB10) warpgroup-MMA shim supports bf16 inputs only. fp16/fp8/int8 " \
        "warpgroup MMA is not implemented on SM120 -- the warp-level mma.sync bases " \
        "in warp.cuh are incomplete for these types. Use bf16, add the missing " \
        "warp::mma_*_base overloads, or call warp-level mma directly. " \
        "See include/ops/group/mma/warpgroup_sm120.cuh.")

// ---------------------------------------------------------------------------
//  Helper: load full B from shared into a per-warp register tile.
//  Each of the 4 warps loads the complete B (group<1> scope).
// ---------------------------------------------------------------------------

// ===========================================================================
//  mma_AB / mm_AB :  D = A @ B   (B is [K, N], loaded col-layout)
// ===========================================================================

// [(register A, shared B) -> register]
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::col> b_reg;
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_AB(d, a, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 0>(d, a, b);
}

// [(shared A, shared B) -> register]
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    // warpgroup-collaborative load: warp w receives its 16*D::height-row band of A.
    rt<T_AB, D::rows, A::cols, ducks::rt_layout::row> a_reg;
    load(a_reg, a); // group<4>::load
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::col> b_reg;
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_AB(d, a_reg, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 0>(d, a, b);
}

// ===========================================================================
//  mma_ABt / mm_ABt :  D = A @ B^T   (B is [N, K], loaded row-layout)
// ===========================================================================

// [(register A, shared B) -> register]
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::row> b_reg;
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_ABt(d, a, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 0>(d, a, b);
}

// [(shared A, shared B) -> register]
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    rt<T_AB, D::rows, A::cols, ducks::rt_layout::row> a_reg;
    load(a_reg, a); // group<4>::load
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::row> b_reg;
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_ABt(d, a_reg, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 0>(d, a, b);
}

// ===========================================================================
//  mma_AtB / mm_AtB :  D = A^T @ B   (A is [K, M] shared, B is [K, N] shared)
//  Warp w owns M-rows [16w,16w+16), i.e. A's column band [.,16w:16w+16).
// ===========================================================================

template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    // warp w's column band of A: rows [0,K), cols [16w, 16w+16) -> col-layout [K,16]
    rt<T_AB, A::rows, D::rows, ducks::rt_layout::col> a_w;
    group<1>::load(a_w, const_cast<A&>(a).template subtile<A::rows, D::rows>({0, (int)warpid()}));
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::col> b_reg; // [K, N] col
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_AtB(d, a_w, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma_AtB<D, A, B, 0>(d, a, b);
}

// ===========================================================================
//  mma_AtBt / mm_AtBt :  D = A^T @ B^T  (A is [K, M] shared, B is [N, K] shared)
// ===========================================================================

template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int accumulate=1>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM120_WG_MMA_DTYPE_CHECK(T_AB);
    rt<T_AB, A::rows, D::rows, ducks::rt_layout::col> a_w; // A col band [K,16]
    group<1>::load(a_w, const_cast<A&>(a).template subtile<A::rows, D::rows>({0, (int)warpid()}));
    rt<T_AB, B::rows, B::cols, ducks::rt_layout::row> b_reg; // [N, K] row
    group<1>::load(b_reg, b);
    if constexpr (!accumulate) group<1>::zero(d);
    group<1>::mma_AtBt(d, a_w, b_reg, d);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma_AtBt<D, A, B, 0>(d, a, b);
}

// ===========================================================================
//  Aliases: mma == mma_AB, dot == mma_ABt   (match warpgroup.cuh)
// ===========================================================================

template<ducks::rt::row_layout D, typename A, ducks::st::all B>
__device__ static inline void mma(D &d, const A &a, const B &b) { mma_AB(d, a, b); }
template<ducks::rt::row_layout D, typename A, ducks::st::all B>
__device__ static inline void mm(D &d, const A &a, const B &b) { mm_AB(d, a, b); }
template<ducks::rt::row_layout D, typename A, ducks::st::all B>
__device__ static inline void dot(D &d, const A &a, const B &b) { mma_ABt(d, a, b); }

#undef KITTENS_SM120_WG_MMA_DTYPE_CHECK
