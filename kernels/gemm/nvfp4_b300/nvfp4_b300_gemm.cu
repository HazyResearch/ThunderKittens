#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_gemm {

template <typename SF_DTYPE>
struct scale_traits;

template <>
struct scale_traits<fp8e8m0> {
    static constexpr int SC_COLS_PER_MMA = 16;
    static constexpr int REAL_BLOCKS_PER_MMA = 3;
    static constexpr int MMAS_PER_GROUP = 4;
};

template <>
struct scale_traits<fp8e4m3> {
    static constexpr int SC_COLS_PER_MMA = 32;
    static constexpr int REAL_BLOCKS_PER_MMA = 6;
    static constexpr int MMAS_PER_GROUP = 2;
};

template <typename _SCALE_DTYPE, bool _APPLY_GLOBAL_SCALE = false>
struct config {
    static_assert(
        std::is_same_v<_SCALE_DTYPE, fp8e4m3> || std::is_same_v<_SCALE_DTYPE, fp8e8m0>,
        "B300 packed-FP4 GEMM supports E4M3 or E8M0 scale factors");

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int CLC_DEPTH = 3;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    // Tuned B300 profile used for the PR #201 benchmark shapes.
    static constexpr int LOAD_PIPE_DEPTH = 2;
    static constexpr int SF_PIPE_DEPTH = 6;
    static constexpr int SF_TMEM_DEPTH = 6;
    static constexpr int EPI_PIPE_DEPTH = 16;
    static constexpr int SUPERGROUP_SIZE = 8;
    static constexpr bool RASTER_ALONG_N = true;
    static constexpr bool APPLY_GLOBAL_SCALE = _APPLY_GLOBAL_SCALE;

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int MMA_PER_TILE = 8;
    static constexpr int Kb = 96 * MMA_PER_TILE;
    static constexpr int B_SC_SIZE = Nb / 128;

    using SCALE_DTYPE = _SCALE_DTYPE;
    using ST = scale_traits<SCALE_DTYPE>;
    static constexpr int SC_COLS_PER_MMA = ST::SC_COLS_PER_MMA;
    static constexpr int REAL_BLOCKS_PER_MMA = ST::REAL_BLOCKS_PER_MMA;

    // Tight K96 scale storage contains only the real block scales. Grouping two E4M3
    // MMAs or four E8M0 MMAs produces three complete 32x16 scale-factor atoms.
    static constexpr int SF_TILE_ROWS = 24;
    static constexpr int SF_SMEM_COLS_PER_MMA = SC_COLS_PER_MMA;
    static constexpr int SF_GROUP_MMAS = ST::MMAS_PER_GROUP;
    static constexpr int SF_GROUPS_PER_TILE = MMA_PER_TILE / SF_GROUP_MMAS;
    static constexpr int SF_GROUP_ROWS = SF_TILE_ROWS * SF_GROUP_MMAS / MMA_PER_TILE;
    static constexpr int SF_ATOMS_PER_GROUP = SF_GROUP_MMAS * REAL_BLOCKS_PER_MMA / 4;
    static constexpr int SF_GROUP_TMEM_COLS = SF_ATOMS_PER_GROUP * 16;

    static_assert(SF_ATOMS_PER_GROUP == 3);
    static_assert(Nb + (SF_GROUP_TMEM_COLS / 4) * SF_TMEM_DEPTH * (1 + B_SC_SIZE) <= MAX_TENSOR_COLS,
                  "scale-factor pipeline exceeds tensor-memory capacity");
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st<typename C::SCALE_DTYPE, C::SF_TILE_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st<typename C::SCALE_DTYPE, C::SF_TILE_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    // Each scale buffer is a contiguous group of K96 MMA scales.
    using A_sc_grp_tile = st<typename C::SCALE_DTYPE, C::SF_GROUP_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    using B_sc_grp_tile = st<typename C::SCALE_DTYPE, C::SF_GROUP_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, false>;
    using scale_atom = st<typename C::SCALE_DTYPE, 32, 16, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    union scale_group {
        A_sc_grp_tile tile;
        scale_atom atoms[C::SF_ATOMS_PER_GROUP];
    };
    static_assert(sizeof(scale_group) == sizeof(A_sc_grp_tile));

    // Fine-grained sub-tile staging of A/B along K. Each sub-load
    // fills exactly one swizzle period (one L2-line-aligned K-slice) of the K-tile,
    // signalling its own mbarrier, so the MMA consumer can begin issuing K96 chunks
    // as soon as the slices they touch have landed -- without waiting for the whole
    // K-tile. fp4e2m1_2 is 1 byte/col, so swizzle_bytes == cols-per-period.
    static constexpr int SUBCOL       = A_fp4x2_tile::swizzle_bytes;
    static constexpr int NUM_SUBLOADS = (C::Kb/2) / SUBCOL;
    static_assert((C::Kb/2) % SUBCOL == 0, "K-tile must be a whole number of swizzle periods.");
    // Explicit swizzle_bytes so the type matches A_fp4x2_tile::subtile<SUBCOL>(s).
    using A_fp4x2_sub_tile = st_fp4e2m1_2<C::Mb/2, SUBCOL, true, SUBCOL>;
    using B_fp4x2_sub_tile = st_fp4e2m1_2<C::Nb/2, SUBCOL, true, SUBCOL>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile, A_fp4x2_sub_tile>;
    using A_sc_gl        = gl<typename C::SCALE_DTYPE, -1, -1, C::SF_TILE_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, A_sc_tile, A_sc_grp_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile, B_fp4x2_sub_tile>;
    using B_sc_gl        = gl<typename C::SCALE_DTYPE, -1, -1, C::SF_TILE_ROWS, C::SF_SMEM_COLS_PER_MMA*C::MMA_PER_TILE, B_sc_tile, B_sc_grp_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;           // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (K // Kb) x 24 x (SC_COLS_PER_MMA*MMA_PER_TILE)
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // M x (N // 2)
    B_sc_gl        B_sc;        // (N // 128) x (K // Kb) x 24 x (SC_COLS_PER_MMA*MMA_PER_TILE)
    B_sc_global_gl B_sc_global; // (1,)
    D_gl           D;           // M x N

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        scale_group A;
        scale_group B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D;
    };
    __host__ inline dim3 grid() const {
        // Full-problem grid in CTA units; cluster-launch-control steals the non-resident
        // remainder, so no host-side persistence sizing is needed.
        return dim3((D.rows() / C::Mb) * (D.cols() / C::Nb) * C::CLUSTER_SIZE);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::SF_PIPE_DEPTH  + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A.template prefetch_tma<typename G::A_fp4x2_sub_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_grp_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_sub_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_grp_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int home_cluster = blockIdx.x / C::CLUSTER_SIZE;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int num_col_blocks = g.D.cols() / C::Nb;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    // A/B tiles ride `stage`/`phasebits` (ring LOAD_PIPE_DEPTH); scales ride the independent
    // `sf_stage`/`sf_phasebits` (ring SF_PIPE_DEPTH). Each producer warp owns its own copies.
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    int sf_stage = 0;
    uint32_t sf_phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::SF_PIPE_DEPTH]   = sm_allocator.allocate<G::input_scales_t, C::SF_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Allocate tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH][G::NUM_SUBLOADS];
    __shared__ semaphore scales_arrived[C::SF_PIPE_DEPTH];
    // A/B tiles and scales now have independent finished-rings (decoupled pipe depths).
    __shared__ semaphore tiles_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_finished[C::SF_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ clc::handle clc_response[C::CLC_DEPTH];
    __shared__ semaphore clc_full[C::CLC_DEPTH];
    __shared__ semaphore clc_empty[C::CLC_DEPTH];
    __shared__ semaphore clc_readers_done[C::CLC_DEPTH];
    __shared__ semaphore clc_initially_armed;
    // Seven warps read each CTA-local response; CTA 0 also has the MMA warp.
    const int clc_local_readers = 7 + (cta_id == 0);
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            #pragma unroll
            for (int s = 0; s < G::NUM_SUBLOADS; ++s)
                init_semaphore(tiles_arrived[i][s], 0, 1);
            init_semaphore(tiles_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::SF_PIPE_DEPTH; ++i) {
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(scales_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(clc_initially_armed, 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < C::CLC_DEPTH; ++i) {
            init_semaphore(clc_full[i], 0, 1);
            init_semaphore(clc_empty[i], 0, C::CLUSTER_SIZE);
            init_semaphore(clc_readers_done[i], 0, clc_local_readers);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    auto tile_coord = [&](int cluster_id) {
        return get_swizzled_2d_idx<C::SUPERGROUP_SIZE, C::RASTER_ALONG_N>(
            num_row_blocks, num_col_blocks, cluster_id);
    };
    // Fetch work item `it` (0-based count of steals); returns false when the grid is drained.
    auto clc_next = [&](int it, int &cluster_id) {
        const int slot = it % C::CLC_DEPTH;
        tma::cluster::wait(clc_full[slot], (it / C::CLC_DEPTH) & 1);
        const auto result = clc::query(clc_response[slot]);
        if (result.success)
            cluster_id = static_cast<int>(result.x) / C::CLUSTER_SIZE;
        return result.success != 0;
    };
    auto clc_done = [&](int it) {
        arrive(clc_readers_done[it % C::CLC_DEPTH]);
    };

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int cluster_id = home_cluster;
            for (int item = 0; ; ++item) {
                const int2 coord = tile_coord(cluster_id);
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(tiles_finished[stage], get_phasebit<1>(phasebits, stage));
                    #pragma unroll
                    for (int s = 0; s < G::NUM_SUBLOADS; ++s) {
                        tma::cluster::load_async(input_tiles[stage].A.template subtile<G::SUBCOL>(s), g.A,
                            {coord.x*2 + cta_id, i*G::NUM_SUBLOADS + s}, tiles_arrived[stage][s], (uint16_t)(1<<cta_id), 0);
                        tma::cluster::load_async(input_tiles[stage].B.template subtile<G::SUBCOL>(s), g.B,
                            {coord.y*2 + cta_id, i*G::NUM_SUBLOADS + s}, tiles_arrived[stage][s], (uint16_t)(1<<cta_id), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
                const bool more = clc_next(item, cluster_id);
                clc_done(item);
                if (!more) break;
            }
        } else if (warp_id == 1) {
            // Cluster-launch-control scheduler: every CTA posts the expected response bytes on
            // its local barrier; cluster rank 0 issues the cancellation request, whose response
            // and completion are multicast to both CTAs.
            // Each CTA aggregates its local readers before one cluster-scope release to rank 0.
            // This protects response reuse without placing a cluster fence on every reader.
            everyone::tma::cluster::wait();
            #pragma unroll
            for (int slot = 0; slot < C::CLC_DEPTH; ++slot)
                tma::cluster::expect_bytes<memory_model::RELAXED>(clc_full[slot], sizeof(clc::handle));
            tma::cluster::arrive<memory_model::RELEASE>(clc_initially_armed, 0, 1);
            if (cta_id == 0)
                tma::cluster::wait(clc_initially_armed, 0);

            for (int item = 0; ; ++item) {
                const int slot = item % C::CLC_DEPTH;
                if (item >= C::CLC_DEPTH) {
                    const int empty_phase = ((item - C::CLC_DEPTH) / C::CLC_DEPTH) & 1;
                    wait(clc_readers_done[slot], empty_phase);
                    tma::cluster::expect_bytes<memory_model::RELAXED>(clc_full[slot], sizeof(clc::handle));
                    tma::cluster::arrive<memory_model::RELEASE>(clc_empty[slot], 0, 1);
                    if (cta_id == 0)
                        tma::cluster::wait(clc_empty[slot], empty_phase);
                }
                if (cta_id == 0)
                    clc::schedule(clc_response[slot], clc_full[slot]);
                int cluster_id;
                const bool more = clc_next(item, cluster_id);
                clc_done(item);
                if (!more) break;
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int cluster_id = home_cluster;
            for (int item = 0; ; ++item) {
                const int2 coord = tile_coord(cluster_id);
                for (int i = 0; i < num_red_blocks; ++i) {
                    // Each K-tile's scales are loaded as SF_GROUPS_PER_TILE separate groups (TMA
                    // sub-column boxes), so the SF ring advances at the finer group granularity.
                    #pragma unroll
                    for (int gj = 0; gj < C::SF_GROUPS_PER_TILE; ++gj) {
                        wait(scales_finished[sf_stage], get_phasebit<1>(phasebits, sf_stage));
                        tma::cluster::load_async(input_scales[sf_stage].A.tile, g.A_sc, {coord.x*2 + cta_id, i, gj, 0}, scales_arrived[sf_stage], uint16_t(1 << cta_id), 0);
                        tma::cluster::load_async(input_scales[sf_stage].B[cta_id].tile, g.B_sc, {coord.y*2 + cta_id, i, gj, 0}, scales_arrived[sf_stage], uint16_t(0b11), 0);
                        update_phasebit<1>(phasebits, sf_stage);
                        sf_stage = ring_advance<C::SF_PIPE_DEPTH>(sf_stage);
                    }
                }
                const bool more = clc_next(item, cluster_id);
                clc_done(item);
                if (!more) break;
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiplies
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SF_GROUP_TMEM_COLS*C::SF_TMEM_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SF_GROUP_TMEM_COLS*C::B_SC_SIZE*C::SF_TMEM_DEPTH>>(
                256 + (C::SF_GROUP_TMEM_COLS / 4) * C::SF_TMEM_DEPTH);
            int cluster_id = home_cluster;
            for (int item = 0; ; ++item) {
                tma::cluster::wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_red_blocks; i++) {
                    // A/B tiles arrive at K-tile granularity (one input_tiles[stage] per i).
                    #pragma unroll
                    for (int s = 0; s < G::NUM_SUBLOADS; ++s)
                        tma::expect_bytes(tiles_arrived[stage][s],
                            2*(sizeof(typename G::A_fp4x2_sub_tile) + sizeof(typename G::B_fp4x2_sub_tile)));
                    const uint32_t tphase = get_phasebit<0>(phasebits, stage);
                    int waited = -1;
                    // Scales stream in finer SF groups (SF_GROUP_MMAS MMAs each). Per group:
                    // wait its scales -> copy SMEM->TMEM ring slot sf_stage -> issue its MMAs ->
                    // release the slot. The A/B K96 slice uses the tile-global chunk index `gc`,
                    // while the SF TMEM panel uses the group-local index `lc`.
                    #pragma unroll
                    for (int gj = 0; gj < C::SF_GROUPS_PER_TILE; ++gj) {
                        tma::expect_bytes(scales_arrived[sf_stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[sf_stage], get_phasebit<0>(sf_phasebits, sf_stage));
                        // SMEM slot is sf_stage (load ring); TMEM slot is the shallower ring.
                        const uint32_t sf_tmem = sf_stage % C::SF_TMEM_DEPTH;
                        #pragma unroll
                        for (int k = 0; k < C::SF_ATOMS_PER_GROUP; ++k) {
                            // Copy this group's three tightly packed 32x16 scale atoms to TMEM.
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                sf_tmem*C::SF_GROUP_TMEM_COLS + k*16);
                            load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[sf_stage].A.atoms[k]);
                            // B keeps both N-halves of each K-atom adjacent in TMEM (B[1] +16 past B[0]).
                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                sf_tmem*C::SF_GROUP_TMEM_COLS*C::B_SC_SIZE + k*C::B_SC_SIZE*16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[sf_stage].B[0].atoms[k]);
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, 16>>(
                                sf_tmem*C::SF_GROUP_TMEM_COLS*C::B_SC_SIZE + k*C::B_SC_SIZE*16 + 16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[sf_stage].B[1].atoms[k]);
                        }
                        auto A_sc_sub = A_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SF_GROUP_TMEM_COLS>>(
                            sf_tmem*C::SF_GROUP_TMEM_COLS);
                        auto B_sc_sub = B_sc_tm.template subtile<tt<typename C::SCALE_DTYPE, MAX_TENSOR_ROWS, C::SF_GROUP_TMEM_COLS*C::B_SC_SIZE>>(
                            sf_tmem*C::SF_GROUP_TMEM_COLS*C::B_SC_SIZE);
                        #pragma unroll
                        for (int lc = 0; lc < C::SF_GROUP_MMAS; ++lc) {
                            const int gc = gj*C::SF_GROUP_MMAS + lc; // tile-global K96 chunk index
                            // K96 chunk gc reads bytes [gc*48, gc*48+48) of the K-tile; wait until
                            // every sub-load slice it touches has landed.
                            const int need = (gc*48 + 47) / G::SUBCOL;
                            while (waited < need) { ++waited; wait(tiles_arrived[stage][waited], tphase); }
                            mma2_ABt_chunk<48, true>(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                               A_sc_sub, B_sc_sub, gc, lc, (i == 0 && gc == 0));
                        }
                        // Release this scale group's slot once its copies + MMAs retire.
                        tensor_commit<2>(scales_finished[sf_stage]);
                        update_phasebit<0>(sf_phasebits, sf_stage);
                        sf_stage = ring_advance<C::SF_PIPE_DEPTH>(sf_stage);
                    }
                    // Release the A/B tile slot once the whole K-tile's MMAs retire.
                    tensor_commit<2>(tiles_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
                const bool more = clc_next(item, cluster_id);
                clc_done(item);
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

        int cluster_id = home_cluster;
        for (int item = 0; ; ++item) {
            const int2 coord = tile_coord(cluster_id);

            // Wait for the last matmul to complete.
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Drain all accumulator slices before releasing TMEM, then pipeline their TMA stores.
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
            warpgroup::tma::cluster::arrive<memory_model::RELAXED>(outputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D, D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D, {coord.x*2 + cta_id, C::EPI_PIPE_DEPTH*coord.y + i});
            }
            update_phasebit<0>(phasebits, 0);
            const bool more = clc_next(item, cluster_id);
            if (laneid() == 0) clc_done(item);
            if (!more) break;
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive<memory_model::RELAXED>(tmem_finished, 1 - cta_id);
            tma::cluster::wait(tmem_finished, 0);
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

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SF=" << (std::is_same_v<typename C::SCALE_DTYPE, fp8e8m0> ? "e8m0" : "e4m3")
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << "\n";

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
    const size_t A_scale_elems = (M / 128) * (K / C::Kb) * C::SF_TILE_ROWS * C::SF_SMEM_COLS_PER_MMA * C::MMA_PER_TILE;
    const size_t B_scale_elems = (N / 128) * (K / C::Kb) * C::SF_TILE_ROWS * C::SF_SMEM_COLS_PER_MMA * C::MMA_PER_TILE;
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
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i * 100 + 3, 0.0f, 256.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i * 100 + 4, 0.0f, 256.0f);
        fill<typename C::SCALE_DTYPE, FillMode::RANDOM>(d_A_sc[i], A_scale_elems, seed + i * 100 + 5, 0.5f, 4.0f);
        fill<typename C::SCALE_DTYPE, FillMode::RANDOM>(d_B_sc[i], B_scale_elems, seed + i * 100 + 6, 0.5f, 4.0f);
        fill<float, FillMode::CONSTANT>(d_A_sc_global[i], 1, 1.0f);
        fill<float, FillMode::CONSTANT>(d_B_sc_global[i], 1, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_nvfp4_gemm<__nv_bfloat16, typename C::SCALE_DTYPE, 96, false>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], d_A_sc_global[0], d_B_sc_global[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl a{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl a_sc{d_A_sc[i], M/128, K/C::Kb, nullptr, nullptr};
        typename G::A_sc_global_gl a_sc_global{d_A_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl b{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl b_sc{d_B_sc[i], N/128, K/C::Kb, nullptr, nullptr};
        typename G::B_sc_global_gl b_sc_global{d_B_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::D_gl d{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{a, a_sc, a_sc_global, b, b_sc, b_sc_global, d});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 10;
    int num_iters = ncu ? 1 : 50;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
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

int main() {
    bool ncu = false;

    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(32768,  5120,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(32768, 13824,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(32768,  5120, 13824, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(75776,  5120,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(75776, 13824,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e4m3>>(75776,  5120, 13824, ncu);

    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(32768,  5120,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(32768, 13824,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(32768,  5120, 13824, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(75776,  5120,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(75776, 13824,  5376, ncu);
    run_benchmark<nvfp4_gemm::config<fp8e8m0>>(75776,  5120, 13824, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"
#include "ATen/Functions.h"

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
        using C = nvfp4_gemm::config<fp8e8m0, true>;
        using G = nvfp4_gemm::globals<C>;

        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, A_sc.size(0), A_sc.size(1), G::A_sc_tile::rows, G::A_sc_tile::cols),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, B_sc.size(0), B_sc.size(1), G::B_sc_tile::rows, G::B_sc_tile::cols),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else if (A_sc.scalar_type() == at::kFloat8_e4m3fn) {
        using C = nvfp4_gemm::config<fp8e4m3, true>;
        using G = nvfp4_gemm::globals<C>;

        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, A_sc.size(0), A_sc.size(1), G::A_sc_tile::rows, G::A_sc_tile::cols),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, B_sc.size(0), B_sc.size(1), G::B_sc_tile::rows, G::B_sc_tile::cols),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else {
        TORCH_CHECK(false, "nvfp4_gemm expected A_sc dtype float8_e8m0fnu or float8_e4m3fn");
    }
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
    m.def("nvfp4_quantize", &nvfp4_quantize_entrypoint);
    m.def("fp32_to_fp4x2", &fp32_to_fp4x2_entrypoint);
    m.def("fp4x2_to_fp32", &fp4x2_to_fp32_entrypoint);
}

#endif
