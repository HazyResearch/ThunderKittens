#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;
static constexpr int NUM_PRODUCERS = 1;
static constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;
static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

template <int _SUPERGROUP_SIZE, int _Mb, int _Nb, int _Kb, int _SMEM_PIPE_DEPTH, int _MMA_PIPE_DEPTH, int _TMEM_PIPE_DEPTH>
struct globals {
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    
    static constexpr int CLUSTER_M = 2*Mb;
    static constexpr int CLUSTER_N = Nb;

    static constexpr int SMEM_PIPE_DEPTH = _SMEM_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = _MMA_PIPE_DEPTH;
    static constexpr int TMEM_PIPE_DEPTH = _TMEM_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int NUM_D_TILES = TMEM_PIPE_DEPTH > 1 ? 2 : 1;

    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb/2, Kb>;
    using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;
    
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(d.rows()/Mb*d.cols()/Nb); }
    __host__ __inline__ dim3 block() { return dim3(NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename G>
__cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ G g) {
    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        g.b.template prefetch_tma<typename G::b_tile>();
        g.d.template prefetch_tma<typename G::d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / G::Kb;

    auto get_tile_idx = [&](int block_idx) -> int2 {
        const int cluster_idx = block_idx / CLUSTER_SIZE;
        const int rblks = g.d.rows() / G::CLUSTER_M;
        const int cblks = g.d.cols() / G::CLUSTER_N;
        const int supergroup_cblks = (cblks/G::SUPERGROUP_SIZE)*G::SUPERGROUP_SIZE;
        const int finalgroup_cblks = cblks-supergroup_cblks;
        const int supergroup_numel = G::SUPERGROUP_SIZE*rblks;
        if (cluster_idx < rblks*supergroup_cblks) {
            const int supergroup_idx = cluster_idx/supergroup_numel;
            const int rblk_idx = (cluster_idx%supergroup_numel)/G::SUPERGROUP_SIZE;
            return { (supergroup_idx&1) ? rblks-rblk_idx-1 : rblk_idx, G::SUPERGROUP_SIZE*supergroup_idx + cluster_idx%G::SUPERGROUP_SIZE };
        } else {
            const int supergroup_idx = cluster_idx/supergroup_numel;
            const int remainder_task_id = cluster_idx - supergroup_cblks*cblks;
            const int rblk_idx = remainder_task_id/finalgroup_cblks;
            return { (supergroup_idx&1) ? rblks-rblk_idx-1 : rblk_idx, supergroup_cblks + remainder_task_id%finalgroup_cblks };
        }
    };

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(G::a_tile) * G::SMEM_PIPE_DEPTH +
                  sizeof(G::b_tile) * G::SMEM_PIPE_DEPTH +
                  sizeof(G::d_tile) * G::NUM_D_TILES <= DYNAMIC_SHARED_MEMORY);
    typename G::a_tile (&a_smem)[G::SMEM_PIPE_DEPTH] = al.allocate<G::a_tile, G::SMEM_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[G::SMEM_PIPE_DEPTH] = al.allocate<G::b_tile, G::SMEM_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[G::NUM_D_TILES]     = al.allocate<G::d_tile, G::NUM_D_TILES>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, G::Mb, G::Nb>;

    __shared__ clc::handle clc_handle[G::CLC_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[G::CLC_PIPE_DEPTH], schedule_finished[G::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[G::SMEM_PIPE_DEPTH], inputs_finished[G::SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[G::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < G::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, 3*CLUSTER_SIZE+1);
        }
        #pragma unroll
        for (int i = 0; i < G::SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();
        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            int input_ring = 0;
            int2 tile_coord = get_tile_idx(blockIdx.x);
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    tma::cluster::load_async(a_smem[input_ring], g.a, {tile_coord.x*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {tile_coord.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                }
                tma::cluster::wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_tile_idx(schedule.x);
                else break;
            }
        } else if (warp::laneid() == 0 && warpgroup::warpid() == 2) {
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    tma::cluster::wait(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], ((task_iter+G::CLC_PIPE_DEPTH)/G::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%G::CLC_PIPE_DEPTH], schedule_arrived[task_iter%G::CLC_PIPE_DEPTH]);
                }
                tma::cluster::expect_bytes(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%G::CLC_PIPE_DEPTH]));
                tma::cluster::wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) {
            d_tt_t d_tt[G::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
                if constexpr(G::Mb == 128)
                    d_tt[i] = tm_alloc.allocate<d_tt_t>(i*G::Nb);
                else
                    d_tt[i] = tm_alloc.allocate<d_tt_t>(0, i*G::Nb);
            }
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                tma::cluster::wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
                tma::cluster::wait(outputs_finished[task_iter%G::MMA_PIPE_DEPTH], ((task_iter+G::MMA_PIPE_DEPTH)/G::MMA_PIPE_DEPTH)%2);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(G::a_tile) + 2*sizeof(G::b_tile));
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    if (idx == 0) mm2_ABt (d_tt[task_iter%G::MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[task_iter%G::MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived);
                if (!schedule.success) break;
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();
        d_tt_t d_tt[G::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            if constexpr(G::Mb == 128) d_tt[i] = tm_alloc.allocate<d_tt_t>(   i*G::Nb);
            else                       d_tt[i] = tm_alloc.allocate<d_tt_t>(0, i*G::Nb);
        }
        int2 tile_coord, next_tile_coord = get_tile_idx(blockIdx.x);
        for(int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            tma::cluster::wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_tile_idx(schedule.x);
            wait(outputs_arrived, task_iter%2);
            rt_bf<G::Mb/4, G::Nb/G::TMEM_PIPE_DEPTH> d_reg[G::TMEM_PIPE_DEPTH];
            #pragma unroll
            for(int i = 0; i < G::TMEM_PIPE_DEPTH; i++) {
                warpgroup::load_async(d_reg[i], d_tt[task_iter%G::MMA_PIPE_DEPTH].template subtile<tt<float, G::Mb, G::Nb/G::TMEM_PIPE_DEPTH>>(0, G::Nb/G::TMEM_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                warpgroup::store(d_smem[i%2], d_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async(g.d, d_smem[i%2], {2*tile_coord.x+cta_rank, G::TMEM_PIPE_DEPTH*tile_coord.y+i});
            }
            warpgroup::tma::store_async_read_wait();
            warpgroup::tma::cluster::arrive(outputs_finished[task_iter%G::MMA_PIPE_DEPTH], 0);
            if (!schedule.success) break;
        }
    }
}

template <typename G>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: SUPERGROUP_SIZE=" << G::SUPERGROUP_SIZE << " Mb=" << G::Mb << " Nb=" << G::Nb << " Kb=" << G::Kb << 
                 " SMEM_PIPE_DEPTH=" << G::SMEM_PIPE_DEPTH << " MMA_PIPE_DEPTH=" << G::MMA_PIPE_DEPTH << " TMEM_PIPE_DEPTH=" << G::TMEM_PIPE_DEPTH << "\n";
    std::cout << "Total number of tasks: " << (M / G::Mb * N / G::Nb) << "\n";
    std::cout << "Number of iterations per task: " << (K / G::Kb) << "\n";

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
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::a_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename G::b_gl Bg{d_B[i], nullptr, nullptr, N, K};
        typename G::d_gl Dg{d_C[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Bg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        kernel<G><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        kernel<G><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
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

    // Template parameters: SUPERGROUP_SIZE, Mb, Nb, Kb, SMEM_PIPE_DEPTH, MMA_PIPE_DEPTH, TMEM_PIPE_DEPTH
    N = 1024;
    run_benchmark<globals<4, 128, 128, 128, 4, 2, 2>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<globals<4, 128, 256, 64, 4, 2, 8>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<globals<4, 128, 256, 64, 5, 2, 2>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<globals<8, 128, 256, 64, 6, 2, 8>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<globals<8, 128, 256, 64, 4, 2, 8>>(N, N, N, ncu);

    return 0;
}
