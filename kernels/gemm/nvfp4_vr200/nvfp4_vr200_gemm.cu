#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

namespace nvfp4_gemm {

template <int _N_PAIR_COUNT, int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE>
struct config {
    static_assert(_N_PAIR_COUNT == 1 || _N_PAIR_COUNT == 2 || _N_PAIR_COUNT == 4, "N_PAIR_COUNT must be 1, 2, or 4");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 12, "LOAD_PIPE_DEPTH must be greater than 0 and at most 12");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int N_PAIR_COUNT = _N_PAIR_COUNT;
    static constexpr bool USE_PREFERRED_CLUSTER = N_PAIR_COUNT > 1;
    static constexpr int M_TILE_COUNT = USE_PREFERRED_CLUSTER ? 2 : 1;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int A_PIPE_SLOTS = USE_PREFERRED_CLUSTER ? M_TILE_COUNT * LOAD_PIPE_DEPTH : 0;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    // Use serpentine traversal only for the preferred-cluster tiers.
    static constexpr bool SERPENTINE = USE_PREFERRED_CLUSTER;

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = 2;

    static constexpr int FP8_VALUES_PER_TMEM_COL = 4 / sizeof(fp8e4m3);
    static constexpr int ACC_TMEM_COLS = M_TILE_COUNT * Nb;
    static constexpr int SINGLE_PAIR_SCALE_TMEM_COLS =
        (16 + 32*MMA_PER_TILE) * LOAD_PIPE_DEPTH / FP8_VALUES_PER_TMEM_COL;
    static constexpr int PREFERRED_SCALE_TMEM_COLS =
        (M_TILE_COUNT*2*16 + 2*64) / FP8_VALUES_PER_TMEM_COL;
    static constexpr int TMEM_COLS = ACC_TMEM_COLS +
        (USE_PREFERRED_CLUSTER ? PREFERRED_SCALE_TMEM_COLS : SINGLE_PAIR_SCALE_TMEM_COLS);
    static constexpr int TMEM_LIMIT =
        USE_PREFERRED_CLUSTER ? MAX_TENSOR_COLS_EXCLUSIVE : MAX_TENSOR_COLS;
    static_assert(TMEM_COLS <= TMEM_LIMIT, "Tensor-memory allocation exceeds allocator capacity");
};

template <typename C>
__device__ inline void get_job_coords(int block_idx, int num_row_blocks, int num_col_blocks,
                                      int &row_block_idx, int &col_block_idx) {
    const int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE, false, C::SERPENTINE>(
        num_row_blocks, num_col_blocks, block_idx / C::N_PAIR_COUNT);
    row_block_idx = tile_coord.x;
    col_block_idx = tile_coord.y * C::N_PAIR_COUNT + block_idx % C::N_PAIR_COUNT;
}

template <typename C>
struct globals {
    using A_fp4x2_tile  = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tma_tile = st_fp8e4m3<8, 256, false>;
    using B_fp4x2_tile  = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tma_tile = st_fp8e4m3<8, 256, false>;
    // Each scale buffer has separate TMA and tensor-memory copy views.
    using A_sc_grp_tile = st_fp8e4m3<128, 16, false>;
    using scale_atom    = st_fp8e4m3<32, 16, false>;
    using D_tile        = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    union scale_group {
        A_sc_tma_tile tma;
        A_sc_grp_tile tile;
        scale_atom atoms[4];
    };
    static_assert(sizeof(scale_group) == sizeof(A_sc_tma_tile));

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<fp8e4m3,    1, -1, -1, 256, A_sc_tma_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<fp8e4m3,    1, -1, -1, 256, B_sc_tma_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;           // M x (K // 2)
    A_sc_gl        A_sc;        // (M // 128) x (K // 32) x 256
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // N x (K // 2)
    B_sc_gl        B_sc;        // (N // 128) x (K // 32) x 256
    B_sc_global_gl B_sc_global; // (1,)
    D_gl           D;           // M x N

    struct ab_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct b_tiles_t {
        B_fp4x2_tile B;
    };
    using input_tiles_t = std::conditional_t<C::USE_PREFERRED_CLUSTER, b_tiles_t, ab_tiles_t>;
    struct input_scales_t {
        scale_group A[C::M_TILE_COUNT];
        scale_group B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::M_TILE_COUNT][C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        const int num_ctas = (D.rows()/(C::M_TILE_COUNT*(C::Mb/2)))*(D.cols()/C::Nb);
        return dim3(C::USE_PREFERRED_CLUSTER ? num_ctas : min(num_ctas, num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(A_fp4x2_tile)   * C::A_PIPE_SLOTS    +
                                               sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= kittens::MAX_SHARED_MEMORY_OVERSIZED - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename TT, typename ST>
__device__ inline void stage_b_scales(TT B_sc_tm, ST &input_scales, int chunk_idx) {
    #pragma unroll
    for (int n = 0; n < 2; n++)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>((j*2+n)*16);
            load_mxnv_scale_async2(B_sc_tm_subtile, input_scales.B[n].atoms[chunk_idx*2 + j]);
        }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tma_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tma_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cluster_cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int cta_id = C::USE_PREFERRED_CLUSTER ? (cluster_cta_id & 1) : cluster_cta_id;
    const int pair_id = cluster_cta_id >> 1;
    const int pair_leader = C::USE_PREFERRED_CLUSTER ? (cluster_cta_id & ~1) : 0;
    const int cluster_width = C::USE_PREFERRED_CLUSTER ? cluster_nctarank() : C::CLUSTER_SIZE;
    const int num_pairs = cluster_width >> 1;
    const uint32_t full_mask = (1u << cluster_width) - 1;
    const uint32_t pair_mask = (0x55555555u & full_mask) << cta_id;
    const uint32_t b_sc_mask = uint32_t(0b11u << pair_leader);
    const int num_row_blocks = g.D.rows() / (C::M_TILE_COUNT * C::Mb);
    const int num_col_blocks = g.D.cols() / (C::N_PAIR_COUNT * C::Nb);
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    const int pair_slot = blockIdx.x >> 1;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
    uint32_t A_stage = 0;
    uint32_t A_phasebits = 0xFFFF0000; // same phase-bit convention for the independent A ring

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::A_fp4x2_tile *A_tiles = nullptr;
    if constexpr (C::USE_PREFERRED_CLUSTER) A_tiles = &sm_allocator.allocate<typename G::A_fp4x2_tile, C::A_PIPE_SLOTS>()[0];
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Allocate tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false, C::USE_PREFERRED_CLUSTER> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle;
    __shared__ semaphore schedule_arrived, schedule_finished;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore A_tiles_arrived[C::USE_PREFERRED_CLUSTER ? C::A_PIPE_SLOTS : 1];
    __shared__ semaphore A_tiles_finished[C::USE_PREFERRED_CLUSTER ? C::A_PIPE_SLOTS : 1];
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished[C::M_TILE_COUNT];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, cluster_width - 1);
        if constexpr (C::USE_PREFERRED_CLUSTER) {
            init_semaphore(schedule_arrived, 0, 1);
            init_semaphore(schedule_finished, 0, (3+C::CONSUMER_WARPGROUPS)*cluster_width + num_pairs);
            #pragma unroll
            for (int i = 0; i < C::A_PIPE_SLOTS; ++i) {
                init_semaphore(A_tiles_arrived[i], 0, 1);
                init_semaphore(A_tiles_finished[i], 0, num_pairs);
            }
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, num_pairs);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::M_TILE_COUNT; ++i) init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // Cluster launch control
    // Elected producer lanes query the result and issue one arrival directly.
    auto clc_next_elected = [&](int it, int &block_idx) {
        wait(schedule_arrived, (it-1)%2);
        const auto schedule = clc::query(clc_handle);
        tma::cluster::arrive(schedule_finished, 0);
        if (!schedule.success) return false;
        block_idx = schedule.x / C::CLUSTER_SIZE + pair_id;
        return true;
    };
    // Consumer warpgroups reconverge before their group leader issues the arrival.
    auto clc_next_warpgroup = [&](int it, int &block_idx, int barrier_id) {
        wait(schedule_arrived, (it-1)%2);
        const auto schedule = clc::query(clc_handle);
        warpgroup::sync(barrier_id);
        warpgroup::tma::cluster::arrive(schedule_finished, 0);
        if (!schedule.success) return false;
        block_idx = schedule.x / C::CLUSTER_SIZE + pair_id;
        return true;
    };

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int block_idx = C::USE_PREFERRED_CLUSTER ? pair_slot : cluster_id;
            for (int it = 0; ; ++it) {
                if constexpr (C::USE_PREFERRED_CLUSTER) {
                    if (it > 0 && !clc_next_elected(it, block_idx)) break;
                } else {
                    if (it > 0) block_idx += gridDim.x / C::CLUSTER_SIZE;
                    if (block_idx >= num_blocks) break;
                }
                int row_block_idx, col_block_idx;
                get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);

                for (int i = 0; i < num_red_blocks; ++i) {
                    if constexpr (C::USE_PREFERRED_CLUSTER) {
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; ++m) {
                            wait(A_tiles_finished[A_stage], get_phasebit<1>(A_phasebits, A_stage));
                            if (pair_id == i % num_pairs)
                                tma::cluster::load_async<dim::ROW, cache_policy::EVICT_LAST>(A_tiles[A_stage], g.A, {row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_id, i}, A_tiles_arrived[A_stage], pair_mask, pair_leader);
                            update_phasebit<1>(A_phasebits, A_stage);
                            A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage);
                        }
                    }
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if constexpr (!C::USE_PREFERRED_CLUSTER) {
                        tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint32_t)(1u<<cluster_cta_id), pair_leader);
                    }
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint32_t)(1u<<cluster_cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            int block_idx = C::USE_PREFERRED_CLUSTER ? pair_slot : cluster_id;
            for (int it = 0; ; ++it) {
                if constexpr (C::USE_PREFERRED_CLUSTER) {
                    if (it > 0 && !clc_next_elected(it, block_idx)) break;
                } else {
                    if (it > 0) block_idx += gridDim.x / C::CLUSTER_SIZE;
                    if (block_idx >= num_blocks) break;
                }
                int row_block_idx, col_block_idx;
                get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if (!C::USE_PREFERRED_CLUSTER || pair_id == i % num_pairs) {
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; ++m)
                            tma::cluster::load_async<dim::ROW, C::USE_PREFERRED_CLUSTER ? cache_policy::EVICT_LAST : cache_policy::NORMAL>(input_scales[stage].A[m].tma, g.A_sc, {row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_id, i, 0}, scales_arrived[stage], C::USE_PREFERRED_CLUSTER ? pair_mask : (uint32_t)(1u<<cluster_cta_id), pair_leader);
                    }
                    tma::cluster::load_async(input_scales[stage].B[cta_id].tma, g.B_sc,
                        {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], b_sc_mask, 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (C::USE_PREFERRED_CLUSTER && warp_id == 1) {
            // Steal pending cluster launches and broadcast the handle to the cluster
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int it = 0; ; ++it) {
                if (cluster_cta_id == 0) {
                    wait(schedule_finished, (it+1)%2);
                    clc::schedule(clc_handle, schedule_arrived);
                }
                tma::expect_bytes(schedule_arrived, sizeof(clc_handle));
                wait(schedule_arrived, it%2);
                auto schedule = clc::query(clc_handle);
                tma::cluster::arrive(schedule_finished, 0);
                if (!schedule.success) break;
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiplies
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            if constexpr (C::M_TILE_COUNT == 1) {
                auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256 + 4*C::LOAD_PIPE_DEPTH);
                for (int it = 0, block_idx = cluster_id; block_idx < num_blocks; ++it, block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    wait(outputs_finished[0], (it+1)%2);
                    tensor_after_thread_sync();
                    for (int i = 0; i < num_red_blocks; i++) {
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[0].tile);
                        #pragma unroll
                        for (int scale_idx = 0; scale_idx < C::MMA_PER_TILE*C::B_SC_SIZE; scale_idx++) {
                            auto slot = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32 + scale_idx*16);
                            load_mxnv_scale_async2(slot, input_scales[stage].B[scale_idx%C::B_SC_SIZE].atoms[scale_idx/C::B_SC_SIZE]);
                        }
                        tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32);
                        if (i == 0) warpgroup::mm2_ABt<64, true>(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm_subtile, B_sc_tm_subtile, inputs_finished[stage]);
                        else        warpgroup::mma2_ABt<64, true>(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm_subtile, B_sc_tm_subtile, inputs_finished[stage]);
                        update_phasebit<0>(phasebits, stage);
                        stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                    }
                    tensor_commit<2>(outputs_arrived);
                }
            } else {
                constexpr int ACC_COLS = C::M_TILE_COUNT * C::Nb;
                full_tt_fl<C::Nb> out_tm[C::M_TILE_COUNT];
                full_tt_fp8e4m3<16> A_sc_tm[C::M_TILE_COUNT][2];
                full_tt_fp8e4m3<64> B_sc_tm[2];
                #pragma unroll
                for (int m = 0; m < C::M_TILE_COUNT; m++) {
                    out_tm[m] = tm_allocator.template allocate<full_tt_fl<C::Nb>>(m*C::Nb);
                    #pragma unroll
                    for (int p = 0; p < 2; p++)
                        A_sc_tm[m][p] = tm_allocator.template allocate<full_tt_fp8e4m3<16>>(ACC_COLS + 8*m + 4*p);
                }
                #pragma unroll
                for (int p = 0; p < 2; p++)
                    B_sc_tm[p] = tm_allocator.template allocate<full_tt_fp8e4m3<64>>(ACC_COLS + 16 + 16*p);
                for (int it = 0; ; ++it) {
                    int block_idx;
                    if (it > 0 && !clc_next_elected(it, block_idx)) break;
                    for (int i = 0; i < num_red_blocks; i++) {
                        const int parity = i & 1;
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; m++)
                            load_mxnv_scale_async2(A_sc_tm[m][parity], input_scales[stage].A[m].tile);
                        stage_b_scales(B_sc_tm[0], input_scales[stage], 0);
                        tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        const int next_A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage);
                        tma::expect_bytes(A_tiles_arrived[A_stage], 2*sizeof(typename G::A_fp4x2_tile));
                        tma::expect_bytes(A_tiles_arrived[next_A_stage], 2*sizeof(typename G::A_fp4x2_tile));
                        wait(A_tiles_arrived[A_stage], get_phasebit<0>(A_phasebits, A_stage));
                        wait(A_tiles_arrived[next_A_stage], get_phasebit<0>(A_phasebits, next_A_stage));
                        st_descriptor<typename G::A_fp4x2_tile, 0> A0_desc(A_tiles[A_stage]);
                        st_descriptor<typename G::A_fp4x2_tile, 0> A1_desc(A_tiles[next_A_stage]);
                        st_descriptor<typename G::B_fp4x2_tile, 0> B_desc(input_tiles[stage].B);
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const bool init = i == 0 && c == 0;
                            const full_tt_fp8e4m3<16> A0_sc_tm(A_sc_tm[0][parity].addr + 2*c);
                            const full_tt_fp8e4m3<16> A1_sc_tm(A_sc_tm[1][parity].addr + 2*c);
                            if (init) { wait(outputs_finished[0], (it+1)%2); tensor_after_thread_sync(); }
                            mma2_ABt_chunk<64, false, true, collector::DISCARD, collector::FILL>(out_tm[0], A0_desc, B_desc, A0_sc_tm, B_sc_tm[c], c, 0, init);
                            if (c == 1) tensor_aread_commit<2>(A_tiles_finished[A_stage], full_mask);
                            if (init) { wait(outputs_finished[1], (it+1)%2); tensor_after_thread_sync(); }
                            mma2_ABt_chunk<64, false, true, collector::DISCARD, collector::LASTUSE>(out_tm[1], A1_desc, B_desc, A1_sc_tm, B_sc_tm[c], c, 0, init);
                            if (c == 1) tensor_aread_commit<2>(A_tiles_finished[next_A_stage], full_mask);
                            if (c == 0) stage_b_scales(B_sc_tm[1], input_scales[stage], 1);
                        }
                        tensor_commit<2>(inputs_finished[stage], full_mask);
                        update_phasebit<0>(A_phasebits, A_stage);
                        update_phasebit<0>(A_phasebits, next_A_stage);
                        A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage, C::M_TILE_COUNT);
                        update_phasebit<0>(phasebits, stage);
                        stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                    }
                    tensor_commit<2>(outputs_arrived, (uint32_t)(0b11u << pair_leader));
                }
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
        pdl::wait();
        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];
        constexpr int EPI_COLS = C::Nb / C::EPI_PIPE_DEPTH;

        int block_idx = C::USE_PREFERRED_CLUSTER ? pair_slot : cluster_id;
        for (int it = 0; ; ++it) {
            if constexpr (C::USE_PREFERRED_CLUSTER) {
                if (it > 0 && !clc_next_warpgroup(it, block_idx, 1)) break;
            } else {
                if (it > 0) block_idx += gridDim.x / C::CLUSTER_SIZE;
                if (block_idx >= num_blocks) break;
            }
            int row_block_idx, col_block_idx;
            get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
            const int d_col_base = C::EPI_PIPE_DEPTH*col_block_idx;

            wait(outputs_arrived, it % 2);

            for (int m = 0; m < C::M_TILE_COUNT; ++m) {
                const auto acc_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(m*C::Nb);
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb/8, EPI_COLS> D_reg;
                    warpgroup::load_async(D_reg, acc_tm.template subtile<full_tt_fl<EPI_COLS>>(0, i*EPI_COLS));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished[m], pair_leader, 1);
                    }
                    warp::mul(D_reg, D_reg, global_scale);
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[m][i%C::NUM_D_TILES], D_reg);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[m][i%C::NUM_D_TILES], {row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_id, d_col_base + i});
                }
            }
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) {
                if constexpr (C::USE_PREFERRED_CLUSTER) {
                    #pragma unroll
                    for (int r = 0; r < C::CLUSTER_SIZE*C::N_PAIR_COUNT; ++r)
                        if (r < cluster_width && r != cluster_cta_id) tma::cluster::arrive(tmem_finished, r);
                } else {
                    tma::cluster::arrive(tmem_finished, 1 - cluster_cta_id);
                }
            }
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

} // namespace nvfp4_gemm

namespace nvfp4_utils {

__global__ void deswizzle_a_scales(__nv_fp8_e4m3 *natural, const __nv_fp8_e4m3 *swizzled, int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * (K / 16)) return;
    int row = idx / (K / 16), k_block_idx = idx % (K / 16);
    natural[(size_t(row/128) * (K/256) + k_block_idx/16) * 2048 + (row%128) * 16 + k_block_idx%16]
        = swizzled[scale_swizzle_idx(row, k_block_idx, K / 16)];
}

} // namespace nvfp4_utils

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

static constexpr double REL_ERR_TOL = 5e-6;

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES
              << " M_TILE_COUNT=" << C::M_TILE_COUNT << " N_PAIR_COUNT=" << C::N_PAIR_COUNT << "\n";

    constexpr size_t M_STEP = C::M_TILE_COUNT * C::Mb;
    constexpr size_t N_STEP = C::N_PAIR_COUNT * C::Nb;
    if (M % M_STEP != 0 || N % N_STEP != 0 || K % C::Kb != 0) {
        std::cout << "unsupported shape: M, N, K must be multiples of " << M_STEP << ", "
                  << N_STEP << ", " << C::Kb << "\n";
        std::exit(EXIT_FAILURE);
    }

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
    std::vector<__nv_fp8_e4m3*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_A_sc_natural(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B_sc(arg_group_count);
    std::vector<float*> d_A_sc_global(arg_group_count);
    std::vector<float*> d_B_sc_global(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_A_sc_natural[i], M*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_A_sc_global[i], sizeof(float));
        cudaMalloc(&d_B_sc_global[i], sizeof(float));
        cudaMalloc(&d_D[i], M * N * sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M * N * sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i * 100, 0.0f, 255.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i * 100 + 1, 0.0f, 255.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A_sc[i], M*K/16, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B_sc[i], N*K/16, seed + i*100 + 3, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_A_sc_global[i], 1, seed + i * 100 + 4, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_B_sc_global[i], 1, seed + i * 100 + 5, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
        nvfp4_utils::deswizzle_a_scales<<<(M*K/16 + 255)/256, 256>>>(d_A_sc_natural[i], d_A_sc[i], M, K);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_nvfp4_gemm<__nv_bfloat16>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], d_A_sc_global[0], d_B_sc_global[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{d_A_sc_natural[i], nullptr, M/128, K/32, nullptr};
        typename G::A_sc_global_gl Asgg{d_A_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{d_B_sc[i], nullptr, N/128, K/32, nullptr};
        typename G::B_sc_global_gl Bsgg{d_B_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Asgg, Bg, Bsg, Bsgg, Dg});
    }

    // Set kernel attributes
    set_oversized_smem(kernel_entrypoint<C>, g[0].dynamic_shared_memory());
    LaunchConfig<true, true> launch_config = C::USE_PREFERRED_CLUSTER
        ? LaunchConfig<true, true>(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0,
                                   dim3(C::CLUSTER_SIZE * C::N_PAIR_COUNT, 1, 1),
                                   dim3(C::CLUSTER_SIZE, 1, 1))
        : LaunchConfig<true, true>(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0,
                                   C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

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
    check_correctness(d_D[0], d_D_ref, M * N, REL_ERR_TOL);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_A_sc_natural[i]);
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

    // Template parameters: N_PAIR_COUNT, LOAD_PIPE_DEPTH, SUPERGROUP_SIZE
    run_benchmark<nvfp4_gemm::config<1, 5, 4>>( 1024,  1024,  1024, ncu);
    run_benchmark<nvfp4_gemm::config<1, 5, 4>>( 2048,  2048,  2048, ncu);
    run_benchmark<nvfp4_gemm::config<1, 5, 4>>( 3072,  3072,  3072, ncu);
    run_benchmark<nvfp4_gemm::config<1, 6, 4>>( 4096,  4096,  4096, ncu);
    run_benchmark<nvfp4_gemm::config<2, 5, 4>>( 5120,  5120,  5120, ncu);
    run_benchmark<nvfp4_gemm::config<2, 5, 4>>( 8192,  8192,  8192, ncu);
    run_benchmark<nvfp4_gemm::config<4, 5, 6>>(16384, 16384, 16384, ncu);
    run_benchmark<nvfp4_gemm::config<4, 5, 6>>(20480, 20480, 20480, ncu);
    run_benchmark<nvfp4_gemm::config<4, 5, 6>>(32768, 32768, 32768, ncu);

    return 0;
}
