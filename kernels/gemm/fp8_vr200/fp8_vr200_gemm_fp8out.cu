#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

namespace fp8_gemm {

template <int _Nb, int _M_TILE_COUNT, int _N_PAIR_COUNT,
          int _SUPERGROUP_SIZE, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Nb == 64 || _Nb == 128 || _Nb == 160 || _Nb == 256,
                  "Nb must be 64, 128, 160, or 256");
    static_assert(_M_TILE_COUNT == 1 || _M_TILE_COUNT == 2, "M_TILE_COUNT must be 1 or 2");
    static_assert(_N_PAIR_COUNT == 1 || _N_PAIR_COUNT == 4, "N_PAIR_COUNT must be 1 or 4");
    static_assert(_SUPERGROUP_SIZE >= 1, "SUPERGROUP_SIZE must be at least 1");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 8, "LOAD_PIPE_DEPTH must be 1-8");
    static_assert(_EPI_PIPE_DEPTH >= 1, "EPI_PIPE_DEPTH must be at least 1");
    static_assert(_Nb % _EPI_PIPE_DEPTH == 0 && (_Nb / _EPI_PIPE_DEPTH) % 32 == 0, "FP8 epilogue slices must be a multiple of 32 columns");

    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 128;

    static constexpr int M_TILE_COUNT = _M_TILE_COUNT;
    static constexpr int N_PAIR_COUNT = _N_PAIR_COUNT;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool SINGLE_TILE = M_TILE_COUNT == 1;
    static constexpr bool USE_PREFERRED_CLUSTER = N_PAIR_COUNT > 1;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = SINGLE_TILE ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH >= 2 && LOAD_PIPE_DEPTH <= 5 ? 4 : 2;
    static constexpr int NUM_DRAIN_SEMS = 2;

    // One fp32 accumulator per M tile per MMA pipeline stage, each Nb tmem columns wide.
    static constexpr int TMEM_COLS = M_TILE_COUNT * MMA_PIPE_DEPTH * Nb;
    static_assert(TMEM_COLS <= MAX_TENSOR_COLS_EXCLUSIVE,
                  "Tensor-memory allocation exceeds allocator capacity");
};

template <typename C>
__device__ inline int2 get_job_coords(int block_idx, int num_row_blocks, int num_col_blocks) {
    const int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE, false>(
        num_row_blocks, num_col_blocks, block_idx / C::N_PAIR_COUNT);
    return {tile_coord.x, tile_coord.y * C::N_PAIR_COUNT + block_idx % C::N_PAIR_COUNT};
}

template <typename C>
struct globals {
    using A_tile = st_fp8e4m3<C::Mb / 2, C::Kb>;
    using B_tile = st_fp8e4m3<C::Nb / 2, C::Kb>;
    using D_tile = st_fp8e4m3<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_gl = gl<fp8e4m3, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
    using D_gl = gl<fp8e4m3, 1, 1, -1, -1, D_tile>;
    using scale_gl = gl<float, 1, 1, 1, 1>;

    A_gl A;
    B_gl B;
    D_gl D;
    scale_gl a_scale, b_scale, d_scale; // (1,) each; d_scale converts compute range to e4m3 range

    struct input_tiles_t {
        A_tile A[C::M_TILE_COUNT];
        B_tile B;
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        const int num_ctas = (D.rows()/(C::M_TILE_COUNT*(C::Mb/2)))*((D.cols()+C::Nb-1)/C::Nb);
        if constexpr (C::USE_PREFERRED_CLUSTER) {
            constexpr int preferred_cluster_size = C::CLUSTER_SIZE * C::N_PAIR_COUNT;
            return dim3(max((min(num_ctas, num_sms()) / preferred_cluster_size) * preferred_cluster_size, preferred_cluster_size));
        }
        return dim3(C::SINGLE_TILE ? num_ctas : min(num_ctas, num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= kittens::MAX_SHARED_MEMORY_OVERSIZED - 1024);
        return _dynamic_shared_memory;
    }
};

// Lane-row epilogue step: tmem load, scale to e4m3, packed 16-byte smem store.
template <int N, ducks::tt::all TM, ducks::st::all ST>
__device__ __forceinline__ void load_scale_store_fp8_lanerow(ST &dst, const TM &src,
                                                             int src_col, int dst_col, float scale) {
    static_assert(N == 16 || N == 32);
    static_assert(std::is_same_v<typename TM::dtype, float>);
    static_assert(std::is_same_v<typename ST::dtype, fp8e4m3>);
    static_assert(TM::rows == 128);
    rv_fl<32*N, naive_l> values;
    warpgroup::load_async(values, src, src_col);
    const float2 scale2 = make_float2(scale, scale);
    const float2 zero2 = make_float2(0.0f, 0.0f);
    const int row = 32 * warpgroup::warpid() + laneid();
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst.data));
    #pragma unroll
    for (int j = 0; j < N; j += 16) {
        uint32_t packed[4];
        #pragma unroll
        for (int q = 0; q < 16; q += 4) {
            const float2 lo = base_ops::fma_AxBtC::op<float2>(make_float2(values[j+q][0], values[j+q+1][0]), scale2, zero2);
            const float2 hi = base_ops::fma_AxBtC::op<float2>(make_float2(values[j+q+2][0], values[j+q+3][0]), scale2, zero2);
            packed[q/4] = __nv_fp8x4_e4m3(make_float4(lo.x, lo.y, hi.x, hi.y)).__x;
        }
        const uint32_t dst_addr = ST::idx(smem, {row, dst_col + j});
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                     "r"(dst_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_tile>();
        g.B.template prefetch_tma<typename G::B_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();

    const int cta_rank = cluster_ctarank();
    const int cta_in_pair = C::USE_PREFERRED_CLUSTER ? (cta_rank & 1) : cta_rank;
    const int pair_id = C::USE_PREFERRED_CLUSTER ? (cta_rank >> 1) : 0;
    const int pair_leader = C::USE_PREFERRED_CLUSTER ? (cta_rank & ~1) : 0;

    const int cluster_id = clusterIdx().x;

    const int cluster_width = C::USE_PREFERRED_CLUSTER ? cluster_nctarank() : C::CLUSTER_SIZE;
    const int num_pairs = cluster_width >> 1;
    const uint32_t full_mask = (1u << cluster_width) - 1;
    const uint32_t pair_mask = (0x55555555u & full_mask) << cta_in_pair;
    const uint32_t self_mask = uint32_t(1u << cta_rank);
    const uint32_t pair_ctas_mask = uint32_t(0b11u << pair_leader);

    const int num_row_blocks = g.D.rows() / (C::M_TILE_COUNT * C::Mb);
    const int num_col_blocks = (g.D.cols() + C::N_PAIR_COUNT*C::Nb - 1) / (C::N_PAIR_COUNT * C::Nb);
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = g.A.cols() / C::Kb;

    // Static pair-slot scheduling is deliberate; CLC measured neutral-to-negative here.
    const int pair_slot = blockIdx.x >> 1;
    const int num_pair_slots = gridDim.x >> 1;
    int num_task_iters = 0;
    if constexpr (C::USE_PREFERRED_CLUSTER) {
        const int num_pair_jobs = C::N_PAIR_COUNT * num_blocks;
        const int full_waves = num_pair_jobs / num_pair_slots;
        const int tail_jobs = num_pair_jobs - full_waves * num_pair_slots;
        num_task_iters = full_waves + (pair_slot < tail_jobs ? 1 : 0);
    }
    uint32_t stage = 0;
    uint32_t bitfield = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<G::outputs_t>();
    tensor_allocator<1, C::CLUSTER_SIZE, false, true> tm_alloc;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished[C::NUM_DRAIN_SEMS];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, cluster_width - 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, num_pairs);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::NUM_DRAIN_SEMS; ++i)
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    auto next_job = [&](int task_iter, int &block_idx) {
        if constexpr (C::USE_PREFERRED_CLUSTER) {
            if (task_iter >= num_task_iters) return false;
            block_idx = pair_slot + task_iter * num_pair_slots;
        } else if constexpr (C::SINGLE_TILE) {
            if (task_iter > 0) return false;
            block_idx = cluster_id;
        } else {
            block_idx = cluster_id + task_iter * (gridDim.x / C::CLUSTER_SIZE);
            if (block_idx >= num_blocks) return false;
        }
        return true;
    };

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) { // Producer group
        const int warp_id = warpgroup::warpid();
        if (warp_id == 3) {
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; ; ++task_iter) {
                int block_idx;
                if (!next_job(task_iter, block_idx)) break;
                const auto [row_block_idx, col_block_idx] = get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks);

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(bitfield, stage));
                    if (pair_id == i % num_pairs) {
                        #pragma unroll
                        for (int t = 0; t < C::M_TILE_COUNT; ++t)
                            tma::cluster::load_async(input_tiles[stage].A[t], g.A, {(row_block_idx*C::M_TILE_COUNT + t)*2 + cta_in_pair, i}, tiles_arrived[stage], pair_mask, pair_leader);
                    }
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_in_pair, i},
                                            tiles_arrived[stage], self_mask, 0);
                    update_phasebit<1>(bitfield, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (cta_in_pair == 0 && warp_id == 0) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            full_tt_fl<C::Nb> d_tt[C::M_TILE_COUNT * C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int p = 0; p < C::M_TILE_COUNT * C::MMA_PIPE_DEPTH; ++p)
                d_tt[p] = tm_alloc.template allocate<full_tt_fl<C::Nb>>(p*C::Nb);
            for (int task_iter = 0; ; ++task_iter) {
                int block_idx;
                if (!next_job(task_iter, block_idx)) break;
                const int p = task_iter % C::MMA_PIPE_DEPTH;
                wait(outputs_finished[p], ((task_iter + C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH) % 2);
                tensor_after_thread_sync();
                auto mma_block = [&](bool init) {
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(bitfield, stage));
                    st_descriptor<typename G::A_tile, 0> A_desc(input_tiles[stage].A[0]);
                    st_descriptor<typename G::B_tile, 0> B0_desc(input_tiles[stage].B);
                    if constexpr (C::M_TILE_COUNT == 2) {
                        st_descriptor<typename G::A_tile, 0> A1_desc(input_tiles[stage].A[1]);
                        if (init) {
                            mma2_ABt_chunk(d_tt[0], A_desc, B0_desc, 0, init);
                            wait(outputs_finished[1], (task_iter + 1) % 2);
                            tensor_after_thread_sync();
                            mma2_ABt_chunk(d_tt[1], A1_desc, B0_desc, 0, init);
                        } else {
                            mma2_ABt_chunk<collector::DISCARD, collector::FILL>(d_tt[0], A_desc, B0_desc, 0, init);
                            mma2_ABt_chunk<collector::DISCARD, collector::LASTUSE>(d_tt[1], A1_desc, B0_desc, 0, init);
                        }
                        mma2_ABt_chunk<collector::DISCARD, collector::FILL>(d_tt[0], A_desc, B0_desc, 1, false);
                        mma2_ABt_chunk<collector::DISCARD, collector::LASTUSE>(d_tt[1], A1_desc, B0_desc, 1, false);
                    } else {
                        mma2_ABt_chunk(d_tt[p], A_desc, B0_desc, 0, init);
                        mma2_ABt_chunk(d_tt[p], A_desc, B0_desc, 1, false);
                    }
                    tensor_commit<2>(inputs_finished[stage], full_mask);
                    update_phasebit<0>(bitfield, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                };
                mma_block(true);
                #pragma unroll 1
                for (int i = 1; i < num_red_blocks; i++) mma_block(false);
                tensor_commit<2>(outputs_arrived, pair_ctas_mask);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) { // Consumer group
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        constexpr int EPI_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        constexpr int LOAD_COLS = 16; // lane-row segment width
        constexpr int STORE_WAIT_COUNT = C::SINGLE_TILE ? 1 : C::NUM_D_TILES - 1;
        constexpr int NSUB = C::M_TILE_COUNT * C::EPI_PIPE_DEPTH;
        full_tt_fl<C::M_TILE_COUNT*C::Nb> d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int p = 0; p < C::MMA_PIPE_DEPTH; ++p)
            d_tt[p] = tm_alloc.template allocate<full_tt_fl<C::M_TILE_COUNT*C::Nb>>(p*C::Nb);
        const float global_scale = g.a_scale[{0}] * g.b_scale[{0}] * g.d_scale[{0}];

        for (int task_iter = 0; ; ++task_iter) {
            int block_idx;
            if (!next_job(task_iter, block_idx)) break;
            const auto [row_block_idx, col_block_idx] = get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks);
            const int p = task_iter % C::MMA_PIPE_DEPTH;

            wait(outputs_arrived, task_iter % 2);

            #pragma unroll
            for (int i = 0; i < NSUB; i++) {
                const int dslot = i % C::NUM_D_TILES;
                if (!C::SINGLE_TILE || i >= 2) {
                    warpgroup::tma::store_async_read_wait<STORE_WAIT_COUNT>();
                    warpgroup::sync(1);
                }
                #pragma unroll
                for (int seg = 0; seg < EPI_COLS; seg += LOAD_COLS)
                    load_scale_store_fp8_lanerow<LOAD_COLS>(output_tiles.D[dslot], d_tt[p], EPI_COLS*i + seg, seg, global_scale);
                if ((!C::SINGLE_TILE && i == NSUB/2 - 1) || i == NSUB - 1) {
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished[!C::SINGLE_TILE ? (i == NSUB - 1) : p], pair_leader, 1);
                }
                warpgroup::sync(1);
                const int mt = i / C::EPI_PIPE_DEPTH;
                const int store_row = (row_block_idx*C::M_TILE_COUNT + mt)*2 + cta_in_pair;
                const int store_col = col_block_idx*C::EPI_PIPE_DEPTH + i % C::EPI_PIPE_DEPTH;
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[dslot], {store_row, store_col});
            }
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) {
                if constexpr (C::USE_PREFERRED_CLUSTER) {
                    #pragma unroll
                    for (int r = 0; r < C::CLUSTER_SIZE*C::N_PAIR_COUNT; ++r)
                        if (r < cluster_width && r != cta_rank) tma::cluster::arrive(tmem_finished, r);
                } else {
                    tma::cluster::arrive(tmem_finished, 1 - cta_rank);
                }
            }
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

} // namespace fp8_gemm

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ fp8_gemm::globals<C> g) {
    fp8_gemm::kernel<C>(g);
}

__global__ void reference_fp8_output_kernel(__nv_fp8_e4m3 *D, const __nv_fp8_e4m3 *A,
                                           const __nv_fp8_e4m3 *B, float sAB, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += (float)A[(size_t)row*K + k] * (float)B[(size_t)col*K + k];
    D[(size_t)row*N + col] = __nv_fp8_e4m3(acc * sAB);
}

template <typename C>
__host__ void run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = fp8_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Nb=" << C::Nb << " M_TILE_COUNT=" << C::M_TILE_COUNT
              << " N_PAIR_COUNT=" << C::N_PAIR_COUNT
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << "\n";

    constexpr size_t M_STEP = C::M_TILE_COUNT * C::Mb;
    if (M % M_STEP != 0 || K < C::Kb || K % C::Kb != 0) {
        std::cout << "unsupported shape: M and K must be multiples of " << M_STEP << " and "
                  << C::Kb << "\n";
        std::exit(EXIT_FAILURE);
    }

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    CUDACHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size = M*K + N*K + M*N;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp8_e4m3*> d_A(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_D(arg_group_count);
    __nv_fp8_e4m3* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K));
        CUDACHECK(cudaMalloc(&d_B[i], N*K));
        CUDACHECK(cudaMalloc(&d_D[i], M*N*sizeof(__nv_fp8_e4m3)));
    }
    CUDACHECK(cudaMalloc(&d_D_ref, M*N*sizeof(__nv_fp8_e4m3)));
    float *d_scales; // {a_scale, b_scale, d_scale}
    CUDACHECK(cudaMalloc(&d_scales, 3*sizeof(float)));

    // Initialize matrices with random values on device; 448 is the max finite e4m3 magnitude.
    const float inv_scale = 448.0f;
    const float s = 1.0f / inv_scale, d = 1.0f;
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -inv_scale, inv_scale);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B[i], N*K, seed + i*100 + 1, -inv_scale, inv_scale);
        fill<__nv_fp8_e4m3, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    const float h_scales[3] = {s, s, d};
    CUDACHECK(cudaMemcpy(d_scales, h_scales, sizeof(h_scales), cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
        reference_fp8_output_kernel<<<grid, block>>>(d_D_ref, d_A[0], d_B[0], s*s*d, M, N, K);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_gl Ag{reinterpret_cast<fp8e4m3*>(d_A[i]), nullptr, nullptr, M, K};
        typename G::B_gl Bg{reinterpret_cast<fp8e4m3*>(d_B[i]), nullptr, nullptr, N, K};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        typename G::scale_gl As{d_scales+0, nullptr, nullptr, nullptr, nullptr};
        typename G::scale_gl Bs{d_scales+1, nullptr, nullptr, nullptr, nullptr};
        typename G::scale_gl Ds{d_scales+2, nullptr, nullptr, nullptr, nullptr};
        g.push_back(G{Ag, Bg, Dg, As, Bs, Ds});
    }

    set_oversized_smem(kernel_entrypoint<C>, g[0].dynamic_shared_memory());
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0,
                                           dim3(C::CLUSTER_SIZE * C::N_PAIR_COUNT, 1, 1),
                                           dim3(C::CLUSTER_SIZE, 1, 1));

    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Cooldown between configurations
    sleep_ms(500);

    auto launch = [&](int n) {
        for (int i = 0; i < n; i++)
            CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[i % arg_group_count]));
    };
    launch(num_warmups);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    launch(num_iters);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    check_correctness(d_D[0], d_D_ref, M * N, 5e-6);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaFree(d_A[i]));
        CUDACHECK(cudaFree(d_B[i]));
        CUDACHECK(cudaFree(d_D[i]));
    }
    CUDACHECK(cudaFree(d_D_ref));
    CUDACHECK(cudaFree(d_scales));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
}

int main(int argc, char **) {
    bool ncu = argc > 1;

    // Template parameters: Nb, M_TILE_COUNT, N_PAIR_COUNT, SUPERGROUP_SIZE, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH
    run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 1>>( 1024,  1024,  1024, ncu);
    run_benchmark<fp8_gemm::config<160, 1, 1, 1, 8, 1>>( 2048,  2048,  2048, ncu);
    run_benchmark<fp8_gemm::config<160, 2, 1, 1, 6, 1>>( 4096,  4096,  4096, ncu);
    run_benchmark<fp8_gemm::config<256, 2, 4, 6, 6, 2>>( 8192,  8192,  8192, ncu);
    run_benchmark<fp8_gemm::config<256, 2, 4, 7, 5, 2>>(16384, 16384, 16384, ncu);
    run_benchmark<fp8_gemm::config<256, 2, 4, 5, 5, 8>>(32768, 32768, 32768, ncu);

    // run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 1>>( 1024,   128,   128, ncu);
    // run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 1>>( 1024,   512,   512, ncu);
    // run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 1>>( 1024,  1024,  1024, ncu);
    // run_benchmark<fp8_gemm::config<128, 1, 1, 1, 8, 2>>( 1024,  2048,  2048, ncu);
    // run_benchmark<fp8_gemm::config<160, 1, 1, 1, 8, 1>>( 1024,  4096,  4096, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 1, 6, 2>>( 1024,  8192,  8192, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 4, 6, 6, 2>>( 1024, 16384, 16384, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 4, 5, 6, 4>>( 1024, 32768, 32768, ncu);

    // run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 1>>( 1024,   384,  3584, ncu);
    // run_benchmark<fp8_gemm::config<128, 1, 1, 1, 8, 1>>( 1024,  3072,  3584, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 1, 6, 2>>( 1024,  9216,  3584, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 1, 6, 2>>( 1024, 12288,  3584, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 8, 6, 2>>( 1024, 24576,  3584, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 8, 6, 4>>( 1024, 36864,  3584, ncu);
    // run_benchmark<fp8_gemm::config<256, 2, 1, 12, 6, 2>>( 1024, 48768,  3584, ncu);

    return 0;
}
