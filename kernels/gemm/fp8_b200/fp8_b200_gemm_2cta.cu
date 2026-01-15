#include "kittens.cuh"
#include "../common.cuh"
#include <iostream>

constexpr int NUM_CONSUMERS = (2); 
constexpr int NUM_PRODUCERS = (1);

using namespace kittens;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 128;

struct matmul_globals {
    using a_tile = st_fp8e4m3<Mb,   Kb>;
    using b_tile = st_fp8e4m3<Nb/2, Kb>;
    using d_tile = st_hf<Mb, 64>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<half,    1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

__device__ static inline int get_iters_per_task(const matmul_globals &g) {
    return g.a.cols() / Kb;
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const matmul_globals &g, int task_iter, bool is_consumer) {
    constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb;
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { 
            (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? (task_id%super_repeat)/SUPER_M : 2*((task_id%super_repeat)/SUPER_M) + ctarank
        };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return {
            (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
        };
    }
    else {
        return { -1, -1 };
    }
}

__global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int iters_per_task = get_iters_per_task(g);

    constexpr int PIPE_DEPTH = 4;

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    using d_tile = matmul_globals::d_tile;
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_DEPTH]                = al.allocate<b_tile, PIPE_DEPTH>();
    d_tile (&d_smem)                            = al.allocate<d_tile>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2); 
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); 
        }
        init_semaphore(outputs_arrived, 0, 1);
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, 2);
        }
    }

    everyone::tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();
        int ctarank = cluster_ctarank(); 
        if(warpgroup::warpid() == 3 && warp::laneid() == 0) {
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) {
                    for(int idx = 0; idx < (PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                        input_ring=ring_advance<PIPE_DEPTH>(input_ring);
                    }
                    if(laneid() == 0) arrive(outputs_arrived); // TODO REVIEW
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); // TODO REVIEW 
                    warp::tma::cluster::load_async(a_smem[input_ring][0], g.a, {(rowcol.x+0), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    warp::tma::cluster::load_async(a_smem[input_ring][1], g.a, {(rowcol.x+1), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    warp::tma::cluster::load_async(b_smem[input_ring],    g.b, { rowcol.y,    idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    input_ring=ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && (warpgroup::warpid() == 0 || warpgroup::warpid() == 1) && warp::laneid() == 0) { // launch the MMA's
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid()*Nb);
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) break;
                tma::cluster::wait(outputs_finished[warpgroup::warpid()], (task_iter+1)%2); // make sure tensor memory is ready to be written to.
                tma::cluster::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=ring_advance<PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroupid*Nb);
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx(g, task_iter, true);
            if(rowcol.x == -1) break;
            kittens::wait(outputs_arrived, task_iter%2);
            rt_hf<Mb/4, d_tile::cols> d_reg[4];
            if(warpgroupid == 1) group<8>::sync(15);
            #pragma unroll
            for(int i = 0; i < Nb/d_tile::cols; i++) {
                warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64*i));
            }
            tensor_load_wait();
            warpgroup::sync(warpgroupid);
            if(warpgroup::laneid() == 0) kittens::warp::tma::cluster::arrive(outputs_finished[warpgroupid], 0); // Tensor memory for warpgroup 0 is now free.
            if(warpgroupid == 0) group<8>::sync(15);
            if(warpgroupid == 1) group<8>::sync(14);
            warpgroup::store(d_smem, d_reg[0]);
            warpgroup::sync(warpgroupid);
            if(warpgroup::warpid() == 0) warp::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+0});
            #pragma unroll
            for(int i = 1; i < Nb/d_tile::cols; i++) {
                tma::store_async_read_wait();
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]);
                warpgroup::sync(warpgroupid);
                if(warpgroup::warpid() == 0) warp::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+i});
            }
            tma::store_async_read_wait();
            if(warpgroupid == 0) group<8>::sync(14);
            group<8>::sync(15); // All consumers sync here.
        }
    }
    everyone::tma::cluster::sync();
}


constexpr bool NCU = false;
#include <iostream>
#include <cuda_bf16.h>


void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, half *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

int run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K + size_t(N) * K + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<fp8e4m3*> d_A(arg_group_count);
    std::vector<fp8e4m3*> d_B(arg_group_count);
    std::vector<half*> d_C(arg_group_count);
    half* d_C_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(fp8e4m3));
        cudaMalloc(&d_B[i], N*K*sizeof(fp8e4m3));
        cudaMalloc(&d_C[i], M*N*sizeof(half));
    }
    cudaMalloc(&d_C_ref, M*N*sizeof(half));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<fp8e4m3, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<fp8e4m3, FillMode::RANDOM>(d_B[i], N*K, seed + i*100 + 1, -1.0f, 1.0f);
        fill<half, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    fill<half, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_gemm<fp8e4m3, half, true>(d_C_ref, d_A[0], d_B[0], M, N, K);
    cudaDeviceSynchronize();

    // Set kernel attributes
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch configuration
    dim3 grid(148, 1);
    dim3 block(NUM_THREADS);

    // Number of iterations
    int num_warmups = NCU ? 0 : 5;
    int num_iters = NCU ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        inner_run(d_A[idx], d_B[idx], d_C[idx], M, N, K, grid, block);
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        inner_run(d_A[idx], d_B[idx], d_C[idx], M, N, K, grid, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_C[0], d_C_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    N = 1024;
    run_benchmark(N, N, N);
    N = 2048;
    run_benchmark(N, N, N);
    N = 4096;
    run_benchmark(N, N, N);
    N = 8192;
    run_benchmark(N, N, N);
    N = 16384;
    run_benchmark(N, N, N);
    return 0;
}