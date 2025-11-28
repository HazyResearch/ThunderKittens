#include "kittens.cuh"
#include <iostream>

using namespace kittens;

constexpr int NUM_CONSUMERS = (1);
constexpr int NUM_PRODUCERS = (1);
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

constexpr int SMEM_PIPE_DEPTH = 4;
constexpr int MMA_PIPE_DEPTH = 2;
constexpr int TMEM_PIPE_DEPTH = 8;
constexpr int CLC_PIPE_DEPTH = 2;
constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

struct matmul_globals {
    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb/2, Kb>;
    using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

__device__ static inline int get_iters_per_task(const matmul_globals &g) {
    return g.a.cols() / Kb;
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const matmul_globals &g, int task_iter, bool is_consumer) {
    constexpr int CLUSTER_M = 2*Mb, CLUSTER_N = Nb;
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { 
            (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M)*2 + ctarank,
            is_consumer ? (task_id%super_repeat)/SUPER_M : 2*((task_id%super_repeat)/SUPER_M) + ctarank
        };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return {
            (super_rows + remainder_id%final_rows)*2 + ctarank,
            is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
        };
    }
    else {
        return { -1, -1 };
    }
}

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int iters_per_task = get_iters_per_task(g);

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    using d_tile = matmul_globals::d_tile;
    
    static_assert(sizeof(a_tile) * SMEM_PIPE_DEPTH +
                  sizeof(b_tile) * SMEM_PIPE_DEPTH +
                  sizeof(d_tile) * 2 <= DYNAMIC_SHARED_MEMORY);
    a_tile (&a_smem)[SMEM_PIPE_DEPTH] = al.allocate<a_tile, SMEM_PIPE_DEPTH>();
    b_tile (&b_smem)[SMEM_PIPE_DEPTH] = al.allocate<b_tile, SMEM_PIPE_DEPTH>();
    d_tile (&d_smem)[2]               = al.allocate<d_tile, 2>();

    everyone::tma::cluster::sync();
    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ int curr_block_idx[CLC_PIPE_DEPTH] = {blockIdx.x, 0};
    __shared__ uint4 clc_handle;
    __shared__ semaphore clc_arrived;
    __shared__ semaphore job_arrived[CLC_PIPE_DEPTH], job_finished[CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[SMEM_PIPE_DEPTH], inputs_finished[SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2); 
            init_semaphore(inputs_finished[i], 0, 1); 
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, 2);
        }
        init_semaphore(clc_arrived, 0, 1);
    }
    everyone::tma::cluster::sync();

    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();
        int ctarank = cluster_ctarank(); 
        if (warp::laneid() == 0 && warpgroup::warpid() == 0) {
            int input_ring = 0; // tracking which input block is being loaded
            for (int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) {
                    for (int idx = 0; idx < (SMEM_PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                        input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                    }
                    arrive(outputs_arrived);
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==SMEM_PIPE_DEPTH-1) arrive(outputs_arrived);
                    tma::cluster::expect(inputs_arrived[input_ring], 0, a_smem[0], b_smem[0]);
                    tma::cluster::load_async(a_smem[input_ring], g.a, {rowcol.x, idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {rowcol.y, idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && warp::laneid() == 0 && warpgroup::warpid() == 1) { // launch the MMA's
            int mma_pipe_stage = 0;
            d_tt_t d_tt[MMA_PIPE_DEPTH] = {tm_alloc.allocate<d_tt_t>(0*Nb), tm_alloc.allocate<d_tt_t>(1*Nb)};
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) break;
                tma::cluster::wait(outputs_finished[mma_pipe_stage], (task_iter/2+1)%2); // make sure tensor memory is ready to be written to.
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt[mma_pipe_stage], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt[mma_pipe_stage], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
                mma_pipe_stage = (mma_pipe_stage + 1) % MMA_PIPE_DEPTH;
            }
        }
        else if(warp::laneid() == 0 && warpgroup::warpid() == 2) { // fetch next block idx
            while (true)
            if (ctarank == 0) {
                // Prevent reordering w.r.t. CLC result write
                asm volatile("{fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;}" ::: "memory");
    
                // Request cancellation of a cluster not launched yet
                // Note: requesting subsequent clusterlaunchcontrol.try_cancel after a failure is undefined
                asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&clc_handle))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&clc_arrived)))
                    : "memory"
                );
                tma::cluster::expect_bytes(clc_arrived, sizeof(clc_handle), ctarank);
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();
        int mma_pipe_stage = 0;
        d_tt_t d_tt[MMA_PIPE_DEPTH] = {tm_alloc.allocate<d_tt_t>(0*Nb), tm_alloc.allocate<d_tt_t>(1*Nb)};
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx(g, task_iter, true);
            if(rowcol.x == -1) break;
            wait(outputs_arrived, task_iter%2);
            #pragma unroll
            for (int i = 0; i < TMEM_PIPE_DEPTH; i++) {
                rt_bf<Mb/WARPGROUP_WARPS, Nb/TMEM_PIPE_DEPTH> d_partial_reg;
                warpgroup::load_async(d_partial_reg, d_tt[mma_pipe_stage].subtile<tt<float, Mb, Nb/TMEM_PIPE_DEPTH>>(0, i*Nb/TMEM_PIPE_DEPTH));
                tensor_load_wait();
                if (warpgroup::laneid()==0) tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                if (i==TMEM_PIPE_DEPTH-1 && warpgroup::laneid()==0) tma::cluster::arrive(outputs_finished[mma_pipe_stage], 0);
                warpgroup::store(d_smem[i % 2], d_partial_reg);
                warpgroup::sync(1);
                if (warpgroup::laneid()==0) tma::store_async(g.d, d_smem[i % 2], {rowcol.x, TMEM_PIPE_DEPTH*rowcol.y+i});
            }
            mma_pipe_stage = (mma_pipe_stage + 1) % MMA_PIPE_DEPTH;
            // rt_bf<Mb/4, d_tile::cols> d_reg[4];
            // if(warpgroupid == 1) group<8>::sync(15);
            // #pragma unroll
            // for(int i = 0; i < Nb/d_tile::cols; i++) {
            //     warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64*i));
            // }
            // tensor_load_wait();
            // warpgroup::sync(warpgroupid);
            // if(warpgroup::laneid() == 0) warp::tma::cluster::arrive(outputs_finished[warpgroupid], 0); // Tensor memory for warpgroup 0 is now free.
            // if(warpgroupid == 0) group<8>::sync(15);
            // if(warpgroupid == 1) group<8>::sync(14);
            // warpgroup::store(d_smem, d_reg[0]);
            // warpgroup::sync(warpgroupid);
            // warpgroup::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+0});
            // #pragma unroll
            // for(int i = 1; i < Nb/d_tile::cols; i++) {
            //     tma::store_async_read_wait();
            //     warpgroup::sync(warpgroupid);
            //     warpgroup::store(d_smem, d_reg[i]);
            //     warpgroup::sync(warpgroupid);
            //     warpgroup::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+i});
            // }
            // tma::store_async_read_wait();
            // if(warpgroupid == 0) group<8>::sync(14);
            // group<8>::sync(15); // All consumers sync here.
        }
    }
}


constexpr bool NCU = false;
constexpr bool CHECK_CORRECTNESS = true;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * N + k];
            }
            c[i * N + j] = sum;
        }
    }
}

void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb<< "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if (CHECK_CORRECTNESS) {
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);

    // Launch kernel
    dim3 grid(148, 1);
    dim3 block(NUM_THREADS);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 500); i++) { // warmup
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    
    cudaEventRecord(start);
    constexpr int ITERS = (NCU ? 1 : 100);
    for(int i = 0; i < ITERS; i++) {
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate duration
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    if (CHECK_CORRECTNESS) {
        // Copy result back to host
        cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

        std::cout << "Copied result back to host" << std::endl;

        // Convert result back to float for comparison
        for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

        std::cout << "Converted result back to float" << std::endl;

        // Check result
        float max_error = 0.0f;
        float average_error = 0.0f;
        int error_count = 0;
        for (int i = 0; i < M * N; ++i) {
            float error = std::abs(h_C[i] - h_C_ref[i]);
            if(error > 1.0) { // large because of bf16 vs fp32 numerics
                if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
                error_count++;
            }
            max_error = std::max(max_error, error);
            average_error += error;
        }
        average_error /= M*N;

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Average error: " << average_error << std::endl;
        std::cout << "Error count: " << error_count << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    // N = 1024;
    // run_benchmark(N, N, N);
    // N = 2048;
    // run_benchmark(N, N, N);
    // N = 4096;
    // run_benchmark(N, N, N);
    N = 4096;
    run_benchmark(N, N, N);
    // N = 16384;
    // run_benchmark(N, N, N);
    return 0;
}
