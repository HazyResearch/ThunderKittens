#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, bool _OVERLAP_MMA_EPI, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 256, "Mb must be 256");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");
    static_assert(_EPI_PIPE_DEPTH >= 1 && _EPI_PIPE_DEPTH <= 16, "EPI_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb; // per MMA
    static constexpr int Nb = _Nb; // per MMA
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool OVERLAP_MMA_EPI = _OVERLAP_MMA_EPI;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = OVERLAP_MMA_EPI ? 1 : 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};

template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb/2, C::Kb>;
    using b_tile = st_bf<C::Nb/2, C::Kb>;
    using d_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(d.rows()/(C::NUM_CONSUMERS*C::Mb/2) * d.cols()/C::Nb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        constexpr size_t _dynamic_shared_memory = sizeof(a_tile) * C::LOAD_PIPE_DEPTH * C::NUM_CONSUMERS +
                                                  sizeof(b_tile) * C::LOAD_PIPE_DEPTH +
                                                  sizeof(d_tile) * C::NUM_D_TILES * C::NUM_CONSUMERS + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        g.b.template prefetch_tma<typename G::b_tile>();
        g.d.template prefetch_tma<typename G::d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / C::Kb;
    const int rblks = g.d.rows() / (C::Mb * C::NUM_CONSUMERS);
    const int cblks = g.d.cols() / C::Nb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS] = al.allocate<G::a_tile, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH]                   = al.allocate<G::b_tile, C::LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[C::NUM_CONSUMERS][C::NUM_D_TILES]     = al.allocate<G::d_tile, C::NUM_CONSUMERS, C::NUM_D_TILES>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb/2, C::Nb>;

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle[C::CLC_PIPE_DEPTH];
    __shared__ semaphore tmem_provisioned, schedule_arrived[C::CLC_PIPE_DEPTH], schedule_finished[C::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH], inputs_finished[C::LOAD_PIPE_DEPTH], outputs_arrived[C::NUM_CONSUMERS], outputs_finished[C::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2+C::NUM_CONSUMERS)*C::CLUSTER_SIZE+C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE*C::NUM_CONSUMERS);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < C::NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], g.a, {(tile_coord.x*2+cta_rank)*C::NUM_CONSUMERS+i, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {tile_coord.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
                else break;
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], ((task_iter+C::CLC_PIPE_DEPTH)/C::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%C::CLC_PIPE_DEPTH], schedule_arrived[task_iter%C::CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%C::CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
                if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::warpid())*C::Nb);
                else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::warpid())*C::Nb);
            }
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                wait(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], ((task_iter+C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH)%2);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::expect_bytes(inputs_arrived[input_ring], (C::CLUSTER_SIZE*C::NUM_CONSUMERS*sizeof(G::a_tile) + 2*sizeof(G::b_tile))/C::NUM_CONSUMERS);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
                if (!schedule.success) break;
            }
        }
    }
    else {
        using epilogue_group = group<WARPGROUP_WARPS*C::NUM_CONSUMERS>;
        if constexpr (!C::OVERLAP_MMA_EPI)
            warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::groupid())*C::Nb);
            else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::groupid())*C::Nb);
        }
        int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
        for(int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            if constexpr (C::OVERLAP_MMA_EPI) {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg;
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::load_async(d_reg, d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        warpgroup::sync(warpgroup::groupid()+1);
                        if (!schedule.success) warpgroup::pdl::arrive();
                        warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            } else {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(d_reg[i], d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::sync(warpgroup::groupid()+1);
                if (!schedule.success) warpgroup::pdl::arrive();
                warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg[i]);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            }
            if (!schedule.success) break;
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) tm_alloc.deprovision();
    }
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " OVERLAP_MMA_EPI=" << C::OVERLAP_MMA_EPI << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << "\n";
    std::cout << "Total number of clusters: " << (M / (C::Mb*C::NUM_CONSUMERS) * N / C::Nb) << "\n";
    std::cout << "Number of iterations per task: " << (K / C::Kb) << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_bfloat16*> d_A(arg_group_count);
    std::vector<__nv_bfloat16*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_C(arg_group_count);
    __nv_bfloat16* d_C_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference GEMM on device
    reference_gemm<__nv_bfloat16, __nv_bfloat16>(d_C_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM on device" << std::endl;

    // Prepare kernel inputs
    std::vector<globals<C>> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename globals<C>::a_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename globals<C>::b_gl Bg{d_B[i], nullptr, nullptr, N, K};
        typename globals<C>::d_gl Dg{d_C[i], nullptr, nullptr, M, N};
        g.push_back(globals<C>{Ag, Bg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Prepare kernel launch configuration
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel<C>, g[idx]);
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

    // Verify results
    check_correctness(d_C[0], d_C_ref, M * N);

    // Clean up
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    int N;
    bool ncu = false;

    // Template parameters: Mb, Nb, Kb, SUPERGROUP_SIZE, OVERLAP_MMA_EPI, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH
    N = 1024;
    run_benchmark<config<256, 64, 128, 4, true, 5, 2>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<config<256, 256, 64, 8, true, 5, 4>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<config<256, 256,  64, 4, false, 4, 8>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<config<256, 256,  64, 8, false, 4, 8>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<config<256, 256,  64, 8, false, 4, 8>>(N, N, N, ncu);

    return 0;
}
