#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_gemm {

template <typename SF_DTYPE>
struct scale_traits;

template <>
struct scale_traits<fp8e8m0> {
    static constexpr int SC_COLS_PER_MMA = 16;
    static constexpr int SF_ATOMS_PER_MMA = 1;
    static constexpr int SCALE_BLOCKS_PER_MMA = 4;
};

template <>
struct scale_traits<fp8e4m3> {
    static constexpr int SC_COLS_PER_MMA = 32;
    static constexpr int SF_ATOMS_PER_MMA = 2;
    static constexpr int SCALE_BLOCKS_PER_MMA = 8;
};

template <
    int _Nb,
    int _LOAD_PIPE_DEPTH,
    int _EPI_PIPE_DEPTH,
    int _SUPERGROUP_SIZE,
    int _NUM_D_TILES,
    bool _OVERLAP_EPI,
    bool _RASTER_ALONG_N=false,
    int _MMA_PER_TILE=8,
    bool _APPLY_GLOBAL_SCALE=true,
    int _PREFERRED_CLUSTER_M=4,
    // Default 2 (not 4): CLC tiling needs (N/Nb) % PREFERRED_CLUSTER_N == 0; 4 crashed on N=13824.
    int _PREFERRED_CLUSTER_N=2,
    typename _SCALE_DTYPE=fp8e8m0
>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5, "LOAD_PIPE_DEPTH must be greater than 0 and at most 5");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    // MMA_PER_TILE % 8 == 0 keeps the fp4 tile on a 128B swizzle atom, required for K96 (see descriptor.cuh).
    static_assert(_MMA_PER_TILE % 8 == 0, "MMA_PER_TILE must be a multiple of 8 (K96 needs a 128B swizzle atom)");

    static constexpr int CLUSTER_SIZE = 2;
    // CUTLASS-style dynamic cluster launch: the driver grants the preferred shape where it can
    // and decomposes the rest into fallback clusters. Work assignment is region-based blockIdx
    // math so both shapes compute identical tiles; only TMA multicast scope differs.
    static constexpr int PREFERRED_CLUSTER_M = _PREFERRED_CLUSTER_M;
    static constexpr int PREFERRED_CLUSTER_N = _PREFERRED_CLUSTER_N;
    static constexpr int FALLBACK_CLUSTER_M = 2;
    static constexpr int FALLBACK_CLUSTER_N = 1;
    static constexpr int REGION_PAIRS_M = PREFERRED_CLUSTER_M / CLUSTER_SIZE;
    static constexpr int REGION_PAIRS_N = PREFERRED_CLUSTER_N;
    static_assert(PREFERRED_CLUSTER_M % CLUSTER_SIZE == 0, "preferred cluster M must contain whole 2CTA MMA groups");
    static_assert(PREFERRED_CLUSTER_M % FALLBACK_CLUSTER_M == 0 && PREFERRED_CLUSTER_N % FALLBACK_CLUSTER_N == 0,
                  "fallback cluster must tile the preferred cluster");
    // Cluster-launch-control work stealing: the grid covers the full problem and resident
    // clusters cancel pending cluster launches to steal their tiles. The response ring must
    // cover the consumption stagger between the load producers and the epilogue.
    static constexpr int CLC_DEPTH = 3;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;
    static constexpr bool RASTER_ALONG_N = _RASTER_ALONG_N;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int MMA_PER_TILE = _MMA_PER_TILE;
    static constexpr bool APPLY_GLOBAL_SCALE = _APPLY_GLOBAL_SCALE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 96 * MMA_PER_TILE;
    static constexpr int B_SC_SIZE = Nb/128;
    using SCALE_DTYPE = _SCALE_DTYPE;
    using ST = scale_traits<SCALE_DTYPE>;
    static constexpr int SC_COLS_PER_MMA = ST::SC_COLS_PER_MMA;
    static constexpr int SF_ATOMS_PER_MMA = ST::SF_ATOMS_PER_MMA;
    static constexpr int SCALE_BLOCKS_PER_MMA = ST::SCALE_BLOCKS_PER_MMA;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st<typename C::SCALE_DTYPE, 32, C::SC_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st<typename C::SCALE_DTYPE, 32, C::SC_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<typename C::SCALE_DTYPE, -1, -1, 32, C::SC_COLS_PER_MMA*C::MMA_PER_TILE, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<typename C::SCALE_DTYPE, -1, -1, 32, C::SC_COLS_PER_MMA*C::MMA_PER_TILE, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;           // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (K // Kb) x 32 x (SC_COLS_PER_MMA*MMA_PER_TILE)
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // M x (N // 2)
    B_sc_gl        B_sc;        // (N // 128) x (K // Kb) x 32 x (SC_COLS_PER_MMA*MMA_PER_TILE)
    B_sc_global_gl B_sc_global; // (1,)
    D_gl           D;           // M x N

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };
    __host__ inline dim3 grid() const {
        // Full-problem grid in CTA units; cluster-launch-control steals the non-resident
        // remainder, so no host-side persistence sizing is needed.
        return dim3(D.rows() / C::Mb * 2, D.cols() / C::Nb);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

__device__ static inline int2 cluster_nctaid() {
    int2 dims;
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(dims.x));
    asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(dims.y));
    return dims;
}

// Fence-free variant of kittens::clc::query. The mbarrier complete-tx wait already orders the
// multicast response's visibility, and clc::query's trailing fence.proxy.async stalls behind
// in-flight TMA traffic (~3% on short-K shapes), so the fence is deliberately omitted here.
__device__ static inline bool clc_query(clc::handle &h, int2 &first_ctaid) {
    uint32_t x, y, z, valid;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b128 clc_result;\n\t"
        "ld.shared.b128 clc_result, [%4];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, clc_result;\n\t"
        "selp.u32 %3, 1, 0, p;\n\t"
        "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"
        "}\n"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(valid)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
        : "memory");
    first_ctaid = {static_cast<int>(x), static_cast<int>(y)};
    return valid != 0;
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
#ifdef NVFP4_DEBUG_CLUSTER_SHAPE
        if (cluster_ctarank() == 0) printf("block (%d,%d): cluster %dx%d\n", blockIdx.x, blockIdx.y, cluster_nctaid().x, cluster_nctaid().y);
#endif
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    // Actual cluster geometry (preferred or fallback), used only for TMA multicast scope,
    // load-issuer election, and barrier signaling.
    const int cta_rank = cluster_ctarank();
    const int2 cluster_dims = cluster_nctaid();
    const int cluster_m = cluster_dims.x;
    const int cluster_n = cluster_dims.y;
    const int cta_x = cta_rank % cluster_m;
    const int cta_y = cta_rank / cluster_m;
    const int cta_id = cta_x & 1;
    const int pair_leader_rank = cta_rank - cta_id;
    const uint16_t pair_mask = uint16_t(0b11u << pair_leader_rank);
    // Work items are cluster-shaped CTA-id chunks: the home chunk comes from blockIdx, and
    // subsequent chunks are stolen via cluster launch control. The effective CTA id maps to
    // a tile through a supergroup swizzle over preferred-cluster-aligned regions, so any mix
    // of preferred and fallback chunks covers the grid bijectively.
    const int regions_x = gridDim.x / C::PREFERRED_CLUSTER_M;
    const int2 home_base{static_cast<int>(blockIdx.x) - cta_x, static_cast<int>(blockIdx.y) - cta_y};
    uint16_t a_mcast_mask = 0;
    uint16_t b_mcast_mask = 0;
    for (int y = 0; y < cluster_n; ++y) {
        a_mcast_mask |= uint16_t(1u << (y * cluster_m + cta_x));
    }
    for (int group_m = 0; group_m < cluster_m / C::CLUSTER_SIZE; ++group_m) {
        b_mcast_mask |= uint16_t(1u << (cta_y * cluster_m + group_m * C::CLUSTER_SIZE + cta_id));
    }
    const uint16_t b_scale_mcast_mask = uint16_t(((1u << cluster_m) - 1u) << (cta_y * cluster_m));
    // Stage-free commits must reach every CTA that issues loads consumed by this pair: the
    // pair's cluster column (A and A scales) and the pair's cluster row (B and B scales).
    // Each CTA then receives cluster_n + cluster_m/2 - 1 commits per stage.
    uint16_t inputs_finished_mask = b_scale_mcast_mask;
    for (int y = 0; y < cluster_n; ++y) {
        inputs_finished_mask |= uint16_t(0b11u << (y * cluster_m + (cta_x & ~1)));
    }
    const int inputs_finished_count = cluster_n + cluster_m / C::CLUSTER_SIZE - 1;
    const int macro_row_blocks = g.D.rows() / C::Mb / C::REGION_PAIRS_M;
    const int macro_col_blocks = g.D.cols() / C::Nb / C::REGION_PAIRS_N;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Allocate tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ clc::handle clc_response[C::CLC_DEPTH];
    __shared__ semaphore clc_full[C::CLC_DEPTH];
    __shared__ semaphore clc_empty[C::CLC_DEPTH];
    // The try_cancel response is multicast to every CTA of the cluster, so slot reuse must be
    // gated cluster-wide: all consumers arrive at rank 0's clc_empty. Per CTA that is the tile
    // and scale producer warps plus the four consumer warps, plus the MMA warp on pair leaders.
    const int cluster_size = cluster_m * cluster_n;
    const int clc_empty_count = cluster_size * 6 + cluster_size / C::CLUSTER_SIZE;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, inputs_finished_count);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < C::CLC_DEPTH; ++i) {
            init_semaphore(clc_full[i], 0, 1);
            init_semaphore(clc_empty[i], 0, clc_empty_count);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    // Map an effective CTA id (home or stolen) to this CTA's output pair tile.
    auto tile_coord = [&](int2 base) -> int2 {
        const int eff_x = base.x + cta_x;
        const int eff_y = base.y + cta_y;
        const int swizzle_region = (eff_x / C::PREFERRED_CLUSTER_M) + (eff_y / C::PREFERRED_CLUSTER_N) * regions_x;
        const int2 macro_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE, C::RASTER_ALONG_N>(
            macro_row_blocks, macro_col_blocks, swizzle_region);
        return {macro_coord.x * C::REGION_PAIRS_M + (eff_x % C::PREFERRED_CLUSTER_M) / C::CLUSTER_SIZE,
                macro_coord.y * C::REGION_PAIRS_N + (eff_y % C::PREFERRED_CLUSTER_N)};
    };
    // Fetch work item `it` (0-based count of steals); returns false when the grid is drained.
    // The caller signals clc_empty[it % CLC_DEPTH] after the response has been read.
    auto clc_next = [&](int it, int2 &base) -> bool {
        const int slot = it % C::CLC_DEPTH;
        wait(clc_full[slot], (it / C::CLC_DEPTH) & 1);
        return clc_query(clc_response[slot], base);
    };

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int2 work_base = home_base;
            for (int item = 0; ; ++item) {
                const int2 coord = tile_coord(work_base);
                for (int i = 0; i < num_red_blocks; ++i) {
#ifdef NVFP4_PERF_PROBE_HOT_LOADS
                    // Perf probe only: always reload K block 0 (results are WRONG, loads L2-hot).
                    const int kb = 0;
#else
                    const int kb = i;
#endif
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if (cta_y == 0) tma::cluster::load_async(input_tiles[stage].A, g.A, {coord.x*2 + cta_id, kb}, tiles_arrived[stage], a_mcast_mask, pair_leader_rank);
                    if (cta_x < C::CLUSTER_SIZE) tma::cluster::load_async(input_tiles[stage].B, g.B, {coord.y*2 + cta_id, kb}, tiles_arrived[stage], b_mcast_mask, pair_leader_rank);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                const bool more = clc_next(item, work_base);
                tma::cluster::arrive(clc_empty[item % C::CLC_DEPTH], 0, 1);
                if (!more) break;
            }
        } else if (warp_id == 1) {
            // Cluster-launch-control scheduler: every CTA posts the expected response bytes on
            // its local barrier; cluster rank 0 issues the cancellation request, whose response
            // and completion are multicast to all CTAs of the (preferred or fallback) cluster.
            // Only rank 0 paces on clc_empty, which collects arrivals from the whole cluster,
            // so a new response never clobbers a slot a peer CTA is still reading.
            everyone::tma::cluster::wait();
            for (int item = 0; ; ++item) {
                const int slot = item % C::CLC_DEPTH;
                if (cta_rank == 0) {
                    if (item >= C::CLC_DEPTH) wait(clc_empty[slot], ((item - C::CLC_DEPTH) / C::CLC_DEPTH) & 1);
                    tma::expect_bytes(clc_full[slot], sizeof(clc::handle));
                    clc::schedule(clc_response[slot], clc_full[slot]);
                } else {
                    tma::expect_bytes(clc_full[slot], sizeof(clc::handle));
                }
                int2 base;
                if (!clc_next(item, base)) break;
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int2 work_base = home_base;
            for (int item = 0; ; ++item) {
                const int2 coord = tile_coord(work_base);
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if (cta_y == 0) tma::cluster::load_async(input_scales[stage].A, g.A_sc, {coord.x*2 + cta_id, i, 0, 0}, scales_arrived[stage], a_mcast_mask, pair_leader_rank);
                    if constexpr (C::B_SC_SIZE == 2) {
                        if (cta_x < C::CLUSTER_SIZE) tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {coord.y*2 + cta_id, i, 0, 0}, scales_arrived[stage], b_scale_mcast_mask, pair_leader_rank);
                    } else if (cta_id == 0 && cta_x < C::CLUSTER_SIZE) {
                        tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {coord.y, i, 0, 0}, scales_arrived[stage], b_scale_mcast_mask, pair_leader_rank);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                const bool more = clc_next(item, work_base);
                tma::cluster::arrive(clc_empty[item % C::CLC_DEPTH], 0, 1);
                if (!more) break;
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiplies
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SC_COLS_PER_MMA*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SC_COLS_PER_MMA*C::B_SC_SIZE*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(
                256 + (C::SC_COLS_PER_MMA / 4) * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            int2 work_base = home_base;
            for (int item = 0; ; ++item) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_red_blocks; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
#ifdef NVFP4_PERF_PROBE_NO_SCALE_CP
                    // Perf probe only: skip steady-state scale TMEM copies (results are WRONG).
                    if (i < C::LOAD_PIPE_DEPTH)
#endif
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        // A K96 MMA covers 96/BLOCK_SIZE K-blocks: 3 for E8M0 (block32), 6 for E4M3
                        // (block16). One 32x16 SF atom holds 4 K-blocks, so that rounds up to
                        // SF_ATOMS_PER_MMA = ceil(blocks/4) atoms (1 for E8M0, 2 for E4M3); copy each.
                        #pragma unroll
                        for (int g = 0; g < C::SF_ATOMS_PER_MMA; ++g) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA + ii*C::SC_COLS_PER_MMA + g*16);
                            auto &A_sc_sm_subtile = *reinterpret_cast<st<typename C::SCALE_DTYPE, 32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + C::SC_COLS_PER_MMA*32*ii + 16*32*g);
                            load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                            // block16 needs both N-halves of a K-atom adjacent in TMEM, so atom g is
                            // at g*SC_COLS_PER_MMA with B[1] +16 past B[0] (E8M0: g==0).
                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE + ii*C::B_SC_SIZE*C::SC_COLS_PER_MMA + g*C::SC_COLS_PER_MMA);
                            auto &B_sc_sm_subtile_0 = *reinterpret_cast<st<typename C::SCALE_DTYPE, 32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + C::SC_COLS_PER_MMA*32*ii + 16*32*g);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                            if constexpr (C::B_SC_SIZE == 2) {
                                auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                    stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE + ii*C::B_SC_SIZE*C::SC_COLS_PER_MMA + g*C::SC_COLS_PER_MMA + 16);
                                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st<typename C::SCALE_DTYPE, 32, 16, false> *>(
                                    reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + C::SC_COLS_PER_MMA*32*ii + 16*32*g);
                                load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                            }
                        }
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt<48>(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::MMA_PER_TILE*C::SC_COLS_PER_MMA>>(stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA),
                                        B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE>>(stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE));
                    else       mma2_ABt<48>(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::MMA_PER_TILE*C::SC_COLS_PER_MMA>>(stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA),
                                        B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE>>(stage*C::MMA_PER_TILE*C::SC_COLS_PER_MMA*C::B_SC_SIZE));
                    // Buffer-release commits must reach every CTA that loaded into this pair's
                    // buffers, not just the pair itself, so commit manually with the full mask.
                    tensor_commit<2>(inputs_finished[stage], inputs_finished_mask);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived, pair_mask);
                update_phasebit<1>(phasebits, 0);
                const bool more = clc_next(item, work_base);
                tma::cluster::arrive(clc_empty[item % C::CLC_DEPTH], 0, 1);
                if (!more) break;
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];

        int2 work_base = home_base;
        for (int item = 0; ; ++item) {
            const int2 coord = tile_coord(work_base);

            // Wait for the last matmul to complete.
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

#ifdef NVFP4_PERF_PROBE_NO_EPI
            // Perf probe only: release the accumulator without draining it (D is WRONG).
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, pair_leader_rank, 1);
#else
            // Load the output from tensor memory into registers and store to HBM.
            if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg;
                    warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished, pair_leader_rank, 1);
                    }
                    if constexpr (C::APPLY_GLOBAL_SCALE) warp::mul(D_reg, D_reg, global_scale);
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {coord.x*2 + cta_id, C::EPI_PIPE_DEPTH*coord.y + i});
                }
            } else {
                rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if constexpr (C::APPLY_GLOBAL_SCALE) warp::mul(D_reg_fl, D_reg_fl, global_scale);
                    warp::copy(D_reg[i], D_reg_fl);
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, pair_leader_rank, 1);
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {coord.x*2 + cta_id, C::EPI_PIPE_DEPTH*coord.y + i});
                }
            }
#endif
            update_phasebit<0>(phasebits, 0);
            const bool more = clc_next(item, work_base);
            if (laneid() == 0) tma::cluster::arrive(clc_empty[item % C::CLC_DEPTH], 0, 1);
            if (!more) break;
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, cta_rank ^ 1);
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

} // namespace nvfp4_gemm

namespace nvfp4_quantize {

struct absmax_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 148 * 4;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct quantize_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;      // This should not change
    static constexpr int TILE_N = 128;      // This should not change
    static constexpr int K_BLOCK_SIZE = 16; // This should not change

    using A_bf16_tile  = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N/2, false>;
    using A_sc_vec     = sv_hf<256>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,      1, -1, -1, 256, A_sc_vec>;
    using A_sc_global_gl = gl<float,     1,  1,  1,  1>;

    A_bf16_gl      A_bf16;      // M x N
    A_fp4x2_gl     A_fp4x2;     // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (N // 64) x 512
    A_sc_global_gl A_sc_global; // (1,)

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_N, A_bf16.rows() / TILE_M);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) + 1024;
    }
};

__global__ void zero_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] = 0.0f;
}

__global__ void absmax_kernel(const globals g) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;
    const size_t numel = g.A_bf16.rows() * g.A_bf16.cols();

    bf16 local_max = __float2bfloat16(0.0f);
    bf16_2 *base_ptr = reinterpret_cast<bf16_2*>(g.A_bf16.raw_ptr);

    for (size_t i = tid; i < numel / 8; i += num_threads) {
        bf16_2 v0, v1, v2, v3;
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(*(uint32_t*)&v0), "=r"(*(uint32_t*)&v1), "=r"(*(uint32_t*)&v2), "=r"(*(uint32_t*)&v3)
            : "l"(base_ptr + i*4)
        );

        bf16_2 abs0 = __habs2(v0);
        bf16_2 abs1 = __habs2(v1);
        bf16_2 abs2 = __habs2(v2);
        bf16_2 abs3 = __habs2(v3);

        bf16_2 max01 = __hmax2(abs0, abs1);
        bf16_2 max23 = __hmax2(abs2, abs3);
        bf16_2 max0123 = __hmax2(max01, max23);

        bf16 curr_max = __hmax(max0123.x, max0123.y);
        local_max = __hmax(local_max, curr_max);
    }

    for (size_t i = (numel / 8) * 8 + tid; i < numel; i += num_threads)
        local_max = __hmax(local_max, __habs(g.A_bf16.raw_ptr[i]));

    #pragma unroll
    for (int offset = WARP_THREADS / 2; offset > 0; offset /= 2) {
        uint32_t local_bits = *reinterpret_cast<unsigned short*>(&local_max);
        uint32_t other_bits = __shfl_xor_sync(0xffffffff, local_bits, offset);
        local_max = __hmax(local_max, *reinterpret_cast<bf16*>(&other_bits));
    }

    __shared__ bf16 shared_max[absmax_config::NUM_WARPS];
    if (laneid() == 0) shared_max[warpid()] = local_max;
    __syncthreads();

    if (warpid() == 0) {
        bf16 val = (laneid() < absmax_config::NUM_WARPS) ? shared_max[laneid()] : __float2bfloat16(0.0f);

        #pragma unroll
        for (int offset = absmax_config::NUM_WARPS / 2; offset > 0; offset /= 2) {
            uint32_t val_bits = *reinterpret_cast<unsigned short*>(&val);
            uint32_t other_bits = __shfl_xor_sync(0xffffffff, val_bits, offset);
            val = __hmax(val, *reinterpret_cast<bf16*>(&other_bits));
        }

        if (laneid() == 0) {
            float val_fl = __bfloat162float(val); // Positive float values keep bit ordering
            atomicMax(reinterpret_cast<uint32_t*>(g.A_sc_global.raw_ptr), *reinterpret_cast<uint32_t*>(&val_fl));
        }
    }
}

__global__ void divide_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] /= 6.0f * 448.0f;
}

template<bool SCALE_2D = false>
__device__ inline void quantize_kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem = *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_vec (&A_sc_smem)[2] = *reinterpret_cast<globals::A_sc_vec(*)[2]>(
        reinterpret_cast<uint64_t>(&A_fp4x2_smem) + sizeof(A_fp4x2_smem));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, A_bf16_smem);
        tma::load_async(A_bf16_smem, G.A_bf16, {row, col}, inputs_arrived);
    }

    // Fetch pre-calculated global scales
    float s_global_dec = G.A_sc_global[{0}];
    float s_global_enc = 1.0f / fmaxf(s_global_dec, 0.000000000001f);

    // We have 128 threads per block. Each thread handles 1 row of 128 elements.
    const int tile_row = tid;
    constexpr int NUM_K_BLOCKS_HALF = globals::TILE_N / globals::K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = globals::K_BLOCK_SIZE / 2;                        // 8
    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK]; // [col_half][k_block][elem]
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];                 // [col_half][k_block]

    // Wait for the inputs to arrive
    __syncthreads();
    wait(inputs_arrived, 0);

    // Load input matrix from shared memory (custom swizzling to avoid bank conflicts)
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                const int offset = (tile_row*globals::TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform NVFP4 quantization
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        // Calculate absolute maximum for each K block
        float amax[NUM_K_BLOCKS_HALF];
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax[k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }

        // For 2D scaling, reduce amax across 16 rows
        if constexpr (SCALE_2D) {
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) {
                #pragma unroll
                for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
                    amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffff, amax[i], mask));
            }
        }

        // Compute the local scales
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax[i] / 6.0f * s_global_enc); // round-to-even

        // Quantize input matrix to FP4 and store to shared memory
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]); // choked
            const float s_enc = 1.0f / fmaxf(s_local_dec*s_global_dec, 0.000000000001f);
            const int offset_base = tile_row*globals::TILE_N/2 + (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*globals::K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int offset = offset_base + ((tid+j)&7);
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_fp4x2_smem)) + offset)
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }
    }

    // Store the scales to shared memory following NVIDIA's scale swizzle layout
    const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;
    asm volatile("{st.shared.b32 [%0], %1;}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem[0])) + scale_offset)
           "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[0][0])));
    asm volatile("{st.shared.b32 [%0], %1;}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem[1])) + scale_offset)
           "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[1][0])));

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp4x2, A_fp4x2_smem, {row, col});
        tma::store_async(G.A_sc, A_sc_smem[0], {row, col*2+0, 0});
        tma::store_async(G.A_sc, A_sc_smem[1], {row, col*2+1, 0});
    }
}

} // namespace nvfp4_quantize

namespace nvfp4_utils {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1024; // arbitrary
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    using A_fp32_gl = gl<float, 1, 1, -1, -1>;
    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1>;

    A_fp32_gl A_fp32;
    A_fp4x2_gl A_fp4x2;
};

__device__ inline void fp32_to_fp4x2_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = {G.A_fp32.raw_ptr[i * 2 + 0], G.A_fp32.raw_ptr[i * 2 + 1]};
        G.A_fp4x2.raw_ptr[i].__x = __nv_cvt_float2_to_fp4x2(A_fp32x2, __NV_E2M1, cudaRoundNearest);
    }
}

__device__ inline void fp4x2_to_fp32_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = static_cast<float2>(G.A_fp4x2.raw_ptr[i]);
        G.A_fp32.raw_ptr[i * 2 + 0] = A_fp32x2.x;
        G.A_fp32.raw_ptr[i * 2 + 1] = A_fp32x2.y;
    }
}

} // namespace nvfp4_utils

#ifndef TORCH_COMPILE

#include "../common.cuh"

// No static __cluster_dims__: cluster shape is chosen at launch (preferred with fallback).
template <typename C>
__launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

template <typename T>
__host__ constexpr const char* scale_dtype_name() {
    if constexpr (std::is_same_v<T, fp8e8m0>) return "e8m0";
    else if constexpr (std::is_same_v<T, fp8e4m3>) return "e4m3";
    else return "unknown";
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SF=" << scale_dtype_name<typename C::SCALE_DTYPE>()
              << " PREF_CLUSTER=" << C::PREFERRED_CLUSTER_M << "x" << C::PREFERRED_CLUSTER_N
              << " RASTER=" << (C::RASTER_ALONG_N ? "along_n" : "along_m")
              << " MMA_PER_TILE=" << C::MMA_PER_TILE
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES
              << " OVERLAP_EPI=" << C::OVERLAP_EPI << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K / 2 + size_t(N) * K / 2 + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp4x2_e2m1*> d_A(arg_group_count);
    std::vector<__nv_fp4x2_e2m1*> d_B(arg_group_count);
    std::vector<typename C::SCALE_DTYPE*> d_A_sc(arg_group_count);
    std::vector<typename C::SCALE_DTYPE*> d_B_sc(arg_group_count);
    std::vector<float*> d_A_sc_global(arg_group_count);
    std::vector<float*> d_B_sc_global(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    const size_t A_scale_elems = (M / 128) * (K / C::Kb) * 32 * C::SC_COLS_PER_MMA * C::MMA_PER_TILE;
    const size_t B_scale_elems = (N / 128) * (K / C::Kb) * 32 * C::SC_COLS_PER_MMA * C::MMA_PER_TILE;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], A_scale_elems*sizeof(typename C::SCALE_DTYPE));
        cudaMalloc(&d_B_sc[i], B_scale_elems*sizeof(typename C::SCALE_DTYPE));
        cudaMalloc(&d_A_sc_global[i], sizeof(float));
        cudaMalloc(&d_B_sc_global[i], sizeof(float));
        cudaMalloc(&d_D[i], M * N * sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M * N * sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2026;
    for (int i = 0; i < arg_group_count; i++) {
        // Random FP4 bytes (every byte is a valid e2m1 pair); unlike a constant fill this exposes K-read bugs.
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i*100 + 3, 0.0f, 256.0f);
        // Random e8m0 scales in [0.5, 4.0] (= 2^-1..2^2) exercise the block-scale path without overflowing bf16.
        fill<typename C::SCALE_DTYPE, FillMode::RANDOM>(d_A_sc[i], A_scale_elems, seed + i*100 + 5, 0.5f, 4.0f);
        fill<typename C::SCALE_DTYPE, FillMode::RANDOM>(d_B_sc[i], B_scale_elems, seed + i*100 + 6, 0.5f, 4.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i*100 + 4, 0.0f, 256.0f);
        fill<float, FillMode::CONSTANT>(d_A_sc_global[i], 1, 1.0f);
        fill<float, FillMode::CONSTANT>(d_B_sc_global[i], 1, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_nvfp4_gemm<__nv_bfloat16, typename C::SCALE_DTYPE, 96, true>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], d_A_sc_global[0], d_B_sc_global[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{d_A_sc[i], M/128, K/C::Kb, nullptr, nullptr};
        typename G::A_sc_global_gl Asgg{d_A_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{d_B_sc[i], N/128, K/C::Kb, nullptr, nullptr};
        typename G::B_sc_global_gl Bsgg{d_B_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Asgg, Bg, Bsg, Bsgg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    dim3 preferred_cluster(C::PREFERRED_CLUSTER_M, C::PREFERRED_CLUSTER_N, 1);
    dim3 fallback_cluster(C::FALLBACK_CLUSTER_M, C::FALLBACK_CLUSTER_N, 1);
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, preferred_cluster, fallback_cluster);

    // Number of iterations
    int num_warmups = ncu ? 0 : 10;
    int num_iters = ncu ? 1 : 50;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]));
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_D[0], d_D_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_A_sc_global[i]);
        cudaFree(d_B[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_B_sc_global[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main(int argc, char **argv) {
    bool ncu = argc > 1 && std::string(argv[1]) == "--ncu";

    // Template parameters: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES,
    //   OVERLAP_EPI, RASTER_ALONG_N, MMA_PER_TILE, APPLY_GLOBAL_SCALE, PREF_M, PREF_N, [SCALE_DTYPE]
    // Note: 2x2 preferred clusters are correct but have shown transient launch failures when the
    // GPU is shared with another process; use 2x1 if the GPU is not exclusive.
    run_benchmark<nvfp4_gemm::config<256, 2, 8, 4, 1, false, true, 8, false, 2, 2, fp8e8m0>>(8192, 8192, 33024, ncu);
    // E4M3/block16 scale panels are 2x the SMEM of E8M0/block32, so this shared (non-subtile) pipe
    // only fits LOAD_PIPE_DEPTH=1 for e4m3.
    run_benchmark<nvfp4_gemm::config<256, 1, 8, 4, 1, false, true, 8, false, 2, 1, fp8e4m3>>(8192, 8192, 3072, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"
#include "ATen/Functions.h"

template <typename C>
void nvfp4_gemm_typed_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using G = nvfp4_gemm::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, A_sc.size(0), A_sc.size(1), 32, G::A_sc_tile::cols),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, B_sc.size(0), B_sc.size(1), 32, G::B_sc_tile::cols),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

using nvfp4_gemm_e8m0_config = nvfp4_gemm::config<256, 2, 16, 8, 2, false, true, 8, false, 2, 1, fp8e8m0>;
// E4M3/block16 scale panels are twice as large as E8M0/block32, so the current shared (non-subtile)
// pipeline fits only one load stage.
using nvfp4_gemm_e4m3_config = nvfp4_gemm::config<256, 1, 16, 8, 2, false, true, 8, false, 2, 1, fp8e4m3>;

void nvfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    if (A_sc.scalar_type() == at::kFloat8_e8m0fnu) {
        nvfp4_gemm_typed_entrypoint<nvfp4_gemm_e8m0_config>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else if (A_sc.scalar_type() == at::kFloat8_e4m3fn) {
        nvfp4_gemm_typed_entrypoint<nvfp4_gemm_e4m3_config>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
    } else {
        TORCH_CHECK(false, "nvfp4_gemm expected A_sc dtype float8_e8m0fnu or float8_e4m3fn");
    }
}

void nvfp4_gemm_e8m0_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    nvfp4_gemm_typed_entrypoint<nvfp4_gemm_e8m0_config>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
}

void nvfp4_gemm_e4m3_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    nvfp4_gemm_typed_entrypoint<nvfp4_gemm_e4m3_config>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D);
}

void nvfp4_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc,
    at::Tensor &A_sc_global,
    bool scale_2d
) {
    using C = nvfp4_quantize::quantize_config;
    using G = nvfp4_quantize::globals;

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<G::A_sc_global_gl>(A_sc_global)
    };

    nvfp4_quantize::zero_kernel<<<1, 1>>>(g);
    nvfp4_quantize::absmax_kernel<<<nvfp4_quantize::absmax_config::NUM_BLOCKS, nvfp4_quantize::absmax_config::NUM_THREADS>>>(g);
    nvfp4_quantize::divide_kernel<<<1, 1>>>(g);
    if (scale_2d) kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<true>>(g);
    else          kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<false>>(g);
}

at::Tensor fp32_to_fp4x2_entrypoint(at::Tensor A_fp32) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp32.options().dtype(at::kFloat4_e2m1fn_x2).requires_grad(false);
    at::Tensor A_fp4x2 = at::empty({A_fp32.size(0), A_fp32.size(1) / 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp32_to_fp4x2_kernel>(g);

    return A_fp4x2;
}

at::Tensor fp4x2_to_fp32_entrypoint(at::Tensor A_fp4x2) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp4x2.options().dtype(at::kFloat).requires_grad(false);
    at::Tensor A_fp32 = at::empty({A_fp4x2.size(0), A_fp4x2.size(1) * 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp4x2_to_fp32_kernel>(g);

    return A_fp32;
}

PYBIND11_MODULE(_C, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm_entrypoint);
    m.def("nvfp4_gemm_e8m0", &nvfp4_gemm_e8m0_entrypoint);
    m.def("nvfp4_gemm_e4m3", &nvfp4_gemm_e4m3_entrypoint);
    m.def("nvfp4_quantize", &nvfp4_quantize_entrypoint);
    m.def("fp32_to_fp4x2", &fp32_to_fp4x2_entrypoint);
    m.def("fp4x2_to_fp32", &fp4x2_to_fp32_entrypoint);
}

#endif
