#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

namespace fp8_gemm {

template <int _Nb, int _M_TILE_COUNT, int _N_PAIR_COUNT, int _SUPERGROUP_SIZE, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Nb == 64 || _Nb == 128 || _Nb == 160 || _Nb == 256,
                  "Nb must be 64, 128, 160, or 256");
    static_assert(_M_TILE_COUNT == 1 || _M_TILE_COUNT == 2, "M_TILE_COUNT must be 1 or 2");
    static_assert(_N_PAIR_COUNT == 1 || _N_PAIR_COUNT == 4, "N_PAIR_COUNT must be 1 or 4");
    static_assert(_SUPERGROUP_SIZE >= 1, "SUPERGROUP_SIZE must be at least 1");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 8, "LOAD_PIPE_DEPTH must be 1-8");
    static_assert(_EPI_PIPE_DEPTH >= 1, "EPI_PIPE_DEPTH must be at least 1");

    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 128;

    static constexpr int M_TILE_COUNT = _M_TILE_COUNT;
    static constexpr int N_PAIR_COUNT = _N_PAIR_COUNT;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool SINGLE_TILE = M_TILE_COUNT == 1;
    static constexpr bool USE_PREFERRED_CLUSTER = N_PAIR_COUNT > 1;
    // Retain serpentine traversal for both the single-pair and preferred-cluster tiers.
    static constexpr bool SERPENTINE = true;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = SINGLE_TILE ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = 2;
    static constexpr bool PER_ACC_DRAIN = M_TILE_COUNT == 2;
    static constexpr int NUM_DRAIN_SEMS = PER_ACC_DRAIN ? 2 : MMA_PIPE_DEPTH;
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
    using A_tile = st_fp8e4m3<C::Mb / 2, C::Kb>;
    using B_tile = st_fp8e4m3<C::Nb / 2, C::Kb>;
    using D_tile = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_gl = gl<fp8e4m3, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
    using D_gl = gl<bf16,    1, 1, -1, -1, D_tile>;

    A_gl A;
    B_gl B;
    D_gl D;

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
            return dim3((min(num_ctas, num_sms()) / preferred_cluster_size) * preferred_cluster_size);
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

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_tile>();
        g.B.template prefetch_tma<typename G::B_tile>();
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
    const int num_row_blocks = g.D.rows() / (C::M_TILE_COUNT * C::Mb);
    const int num_col_blocks = (g.D.cols() + C::N_PAIR_COUNT*C::Nb - 1) / (C::N_PAIR_COUNT * C::Nb);
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = g.A.cols() / C::Kb;
    // Static pair-slot scheduling is deliberate: CLC was neutral at 16K/32K
    // and regressed the 8K preferred-cluster configuration.
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
    uint32_t phasebits = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t      &output_tiles                       = sm_allocator.allocate<G::outputs_t>();

    // Set up mbarriers
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

    auto get_block_idx = [&](int task_iter, int &block_idx) {
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

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; ; ++task_iter) {
                int block_idx;
                if (!get_block_idx(task_iter, block_idx)) break;
                int row_block_idx, col_block_idx;
                get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if constexpr (C::USE_PREFERRED_CLUSTER) {
                        if (pair_id == i % num_pairs) {
                            #pragma unroll
                            for (int t = 0; t < C::M_TILE_COUNT; ++t)
                                tma::cluster::load_async(input_tiles[stage].A[t], g.A, {(row_block_idx*C::M_TILE_COUNT + t)*2 + cta_id, i}, tiles_arrived[stage], pair_mask, pair_leader);
                        }
                    } else {
                        #pragma unroll
                        for (int t = 0; t < C::M_TILE_COUNT; ++t)
                            tma::cluster::load_async(input_tiles[stage].A[t], g.A, {(row_block_idx*C::M_TILE_COUNT + t)*2 + cta_id, i}, tiles_arrived[stage], (uint32_t)(1u<<cluster_cta_id), pair_leader);
                    }
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint32_t)(1u<<cluster_cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiplies
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            // Access tensor memory provisioned by the consumer group.
            tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc;
            tm_alloc.set_addr(tmem_addr);
            full_tt_fl<C::Nb> d_tt[C::M_TILE_COUNT * C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int p = 0; p < C::M_TILE_COUNT * C::MMA_PIPE_DEPTH; ++p)
                d_tt[p] = tm_alloc.template allocate<full_tt_fl<C::Nb>>(p*C::Nb);
            for (int task_iter = 0; ; ++task_iter) {
                int block_idx;
                if (!get_block_idx(task_iter, block_idx)) break;
                const int p = task_iter % C::MMA_PIPE_DEPTH;
                wait(outputs_finished[p], ((task_iter+C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH)%2);
                tensor_after_thread_sync();
                // Peel initialization at compile time so steady-state blocks contain only accumulating MMAs.
                auto mma_block = [&]<bool INIT>(int i) {
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    st_descriptor<typename G::A_tile, 0> A_desc(input_tiles[stage].A[0]);
                    st_descriptor<typename G::B_tile, 0> B0_desc(input_tiles[stage].B);
                    if constexpr (C::M_TILE_COUNT == 2) {
                        st_descriptor<typename G::A_tile, 0> A1_desc(input_tiles[stage].A[1]);
                        if (i == 0) {
                            mma2_ABt_chunk(d_tt[0], A_desc, B0_desc, 0, INIT);
                            wait(outputs_finished[1], (task_iter+1)%2); tensor_after_thread_sync();
                            mma2_ABt_chunk(d_tt[1], A1_desc, B0_desc, 0, INIT);
                        } else {
                            mma2_ABt_chunk<collector::DISCARD, collector::FILL>(d_tt[0], A_desc, B0_desc, 0, INIT);
                            mma2_ABt_chunk<collector::DISCARD, collector::LASTUSE>(d_tt[1], A1_desc, B0_desc, 0, INIT);
                        }
                        mma2_ABt_chunk<collector::DISCARD, collector::FILL>(d_tt[0], A_desc, B0_desc, 1, false);
                        mma2_ABt_chunk<collector::DISCARD, collector::LASTUSE>(d_tt[1], A1_desc, B0_desc, 1, false);
                    } else {
                        mma2_ABt_chunk(d_tt[p], A_desc, B0_desc, 0, INIT);
                        mma2_ABt_chunk(d_tt[p], A_desc, B0_desc, 1, false);
                    }
                    tensor_commit<2>(inputs_finished[stage], full_mask);
                    update_phasebit<0>(phasebits, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                };
                if (num_red_blocks > 0) mma_block.template operator()<true>(0);
                for (int i = 1; i < num_red_blocks; i++) mma_block.template operator()<false>(i);
                tensor_commit<2>(outputs_arrived, (uint32_t)(0b11u << pair_leader));
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group
        everyone::tma::cluster::wait_aligned();
        // Allocate tensor memory
        tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc;
        if (warpgroup::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        constexpr int EPI_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        constexpr int NSUB = C::M_TILE_COUNT * C::EPI_PIPE_DEPTH;
        full_tt_fl<C::M_TILE_COUNT*C::Nb> d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int p = 0; p < C::MMA_PIPE_DEPTH; ++p)
            d_tt[p] = tm_alloc.template allocate<full_tt_fl<C::M_TILE_COUNT*C::Nb>>(p*C::Nb);
        for (int task_iter = 0; ; ++task_iter) {
            int block_idx;
            if (!get_block_idx(task_iter, block_idx)) break;
            int row_block_idx, col_block_idx;
            get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
            const int p = task_iter % C::MMA_PIPE_DEPTH;

            wait(outputs_arrived, task_iter % 2);

            #pragma unroll
            for (int i = 0; i < NSUB; i++) {
                rt_fl<C::Mb / 8, EPI_COLS> d_reg;
                warpgroup::load_async(d_reg, d_tt[p].template subtile<full_tt_fl<EPI_COLS>>(0, EPI_COLS*i));
                if ((C::PER_ACC_DRAIN && i == NSUB/2 - 1) || i == NSUB - 1) {
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished[C::PER_ACC_DRAIN ? (i == NSUB - 1) : p], pair_leader, 1);
                }
                rt_bf<C::Mb / 8, EPI_COLS> d_bf;
                warp::copy(d_bf, d_reg);
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                const int dslot = i % C::NUM_D_TILES;
                warpgroup::store(output_tiles.D[dslot], d_bf);
                warpgroup::sync(1);
                if constexpr (C::M_TILE_COUNT == 1) {
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[dslot], {row_block_idx*2 + cta_id, NSUB*col_block_idx + i});
                } else {
                    const int mt = i / C::EPI_PIPE_DEPTH;
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[dslot], {(row_block_idx*C::M_TILE_COUNT + mt)*2 + cta_id, col_block_idx*C::EPI_PIPE_DEPTH + i % C::EPI_PIPE_DEPTH});
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

static constexpr double REL_ERR_TOL = 5e-6;

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = fp8_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Nb=" << C::Nb << " M_TILE_COUNT=" << C::M_TILE_COUNT
              << " N_PAIR_COUNT=" << C::N_PAIR_COUNT
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << "\n";

    constexpr size_t M_STEP = C::M_TILE_COUNT * C::Mb;
    if (M % M_STEP != 0 || K % C::Kb != 0) {
        std::cout << "unsupported shape: M and K must be multiples of " << M_STEP << " and "
                  << C::Kb << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = M*K + N*K + M*N*2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp8_e4m3*> d_A(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K);
        cudaMalloc(&d_B[i], N*K);
        cudaMalloc(&d_D[i], M*N*sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M*N*sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B[i], N*K, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_gemm<__nv_fp8_e4m3, __nv_bfloat16>(d_D_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_gl Ag{reinterpret_cast<fp8e4m3*>(d_A[i]), nullptr, nullptr, M, K};
        typename G::B_gl Bg{reinterpret_cast<fp8e4m3*>(d_B[i]), nullptr, nullptr, N, K};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Bg, Dg});
    }

    // Set kernel attributes and prepare launch configuration
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
        cudaFree(d_B[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    bool ncu = false;

    // Template parameters: Nb, M_TILE_COUNT, N_PAIR_COUNT, SUPERGROUP_SIZE, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH
    N = 1024;
    run_benchmark<fp8_gemm::config<64, 1, 1, 1, 8, 4>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<fp8_gemm::config<160, 1, 1, 1, 6, 5>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<fp8_gemm::config<160, 2, 1, 1, 7, 5>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<fp8_gemm::config<256, 2, 4, 6, 6, 8>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<fp8_gemm::config<256, 2, 4, 4, 5, 8>>(N, N, N, ncu);
    N = 32768;
    run_benchmark<fp8_gemm::config<256, 2, 4, 4, 5, 8>>(N, N, N, ncu);

    return 0;
}
