#include "kittens.cuh"
#include "prototype.cuh"

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int NUM_CONSUMERS = (2); 
constexpr int NUM_PRODUCERS = (1);

using namespace kittens;
namespace cg = cooperative_groups;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

struct matmul_globals {
    using a_tile = st_fl4_e2m1<Mb, Kb>;
    using b_tile = st_fl4_e2m1<Nb/2, Kb>;
    using d_tile = st_hf<Mb, Nb>;

    using a_gl = gl<fp4e2m1, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp4e2m1, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<half, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

__global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();

    constexpr int PIPE_DEPTH = 4;

    using a_tile = st_fl4_e2m1<Mb, Kb>;
    using b_tile = st_fl4_e2m1<Nb/2, Kb>;
    using d_tile = st_hf<Mb, Nb>;
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_DEPTH] = al.allocate<b_tile, PIPE_DEPTH>();
    d_tile (*d_smem) = reinterpret_cast<d_tile*>(&a_smem[0][0]);

    tma::cluster::sync();
    int ctarank = cluster_ctarank();
    auto all_tmem = allocate_tmem<1, 2>();
    using d_tmem_t = tmem<half, Mb, Nb>;

    d_tmem_t d_tmem = all_tmem.subtile<d_tmem_t>(0, warpgroupid*Nb);

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH];
    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2); 
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); 
        }
    }

    int3 cluster_idx = clusterIdx();
    int row_idx = (warpgroupid == NUM_CONSUMERS) ? cluster_idx.x*4 + ctarank*2 : cluster_idx.x*4 + ctarank*2 + warpgroupid; // units of 128
    int col_idx = cluster_idx.y; // units of 256

    tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS) {

        if ((threadIdx.x % (kittens::WARP_THREADS)) == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("b-first, %d \n", warpgroup::warpid());
        }

        warpgroup::decrease_registers<32>();

        if(warpid == NUM_WORKERS-4) {
            for (auto idx = 0; idx < g.a.cols / Kb; idx++) {
                tma::cluster::expect(inputs_arrived[idx%PIPE_DEPTH], 0, a_smem[0][0], a_smem[0][1], b_smem[0]);
                tma::cluster::load_async(a_smem[idx%PIPE_DEPTH][0], g.a, {row_idx,           idx}, inputs_arrived[idx%PIPE_DEPTH], (uint16_t)(1<<ctarank), 0);
                tma::cluster::load_async(a_smem[idx%PIPE_DEPTH][1], g.a, {(row_idx+1),       idx}, inputs_arrived[idx%PIPE_DEPTH], (uint16_t)(1<<ctarank), 0);
                tma::cluster::load_async(b_smem[idx%PIPE_DEPTH],    g.b, {2*col_idx+ctarank, idx}, inputs_arrived[idx%PIPE_DEPTH], (uint16_t)(1<<ctarank), 0);
                if(idx >= PIPE_DEPTH-1) {
                    tma::cluster::wait(inputs_finished[(idx-PIPE_DEPTH+1)%PIPE_DEPTH], ((idx-PIPE_DEPTH+1)/PIPE_DEPTH)%2);
                }
            }
        }
        tma::cluster::sync();

        if ((threadIdx.x % (kittens::WARP_THREADS)) == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("e-first, %d \n", warpgroup::warpid());
        }
    }
    else {

        if ((threadIdx.x % (kittens::WARP_THREADS)) == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("b-second, %d \n", warpgroup::warpid());
        }

        warpgroup::increase_registers<224>();
        if(ctarank == 0 && warpgroup::warpid() == 0) {
            tma::cluster::wait(inputs_arrived[0], 0);
            mm2_ABt(d_tmem, a_smem[0][warpgroupid], b_smem[0], inputs_finished[0]);
            int idx = 1;
            while(idx < g.a.cols / Kb) {
                tma::cluster::wait(inputs_arrived[idx%PIPE_DEPTH], (idx/PIPE_DEPTH)%2);
                mma2_ABt(d_tmem, a_smem[idx%PIPE_DEPTH][warpgroupid], b_smem[idx%PIPE_DEPTH], inputs_finished[idx%PIPE_DEPTH]);
                idx++;
            }
            tma::cluster::wait(inputs_finished[(idx-1)%PIPE_DEPTH], ((idx-1)/PIPE_DEPTH)%2);
        }
        if ((threadIdx.x % (kittens::WARP_THREADS)) == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("sync, %d \n", warpgroup::warpid());
        }
        tma::cluster::sync();
        rt_hf<Mb/4, Nb> d_reg;
        warpgroup::load_async(d_reg, d_tmem);
        tm_load_wait();
        warpgroup::store(d_smem[warpgroupid], d_reg); 
        warpgroup::sync(warpgroupid);
        if (warpgroup::warpid() == 0) {
            tma::store_async(g.d, d_smem[warpgroupid], {row_idx, col_idx});
        }
        tma::store_async_wait();
    }
}


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
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

void inner_run(fp4e2m1 *d_A, fp4e2m1 *d_B, half *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    std::cerr << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")" << std::endl;
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

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    fp4e2m1 *d_A, *d_B;
    half *d_C;
    cudaMalloc(&d_A, M*K*sizeof(fp4e2m1));
    cudaMalloc(&d_B, K*N*sizeof(fp4e2m1));
    cudaMalloc(&d_C, M*N*sizeof(half));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    fp4e2m1 *h_A_fp4 = new __nv_fp4_e2m1[M * K];
    fp4e2m1 *h_B_fp4 = new __nv_fp4_e2m1[K * N];
    for (int i = 0; i < M * K; ++i) h_A_fp4[i] = fp4e2m1(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp4[i] = fp4e2m1(h_B[i]);
    for (int i = 0; i < M * K; ++i) h_A[i] = float(h_A_fp4[i]);
    for (int i = 0; i < K * N; ++i) h_B[i] = float(h_B_fp4[i]);

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    cudaMemcpy(d_A, h_A_fp4, M*K*sizeof(fp4e2m1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp4, K*N*sizeof(fp4e2m1), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(M / (2*Mb), N / Nb, 1);
    dim3 block(NUM_THREADS);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 1); i++) { // warmup
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 5);
    for(int i = 0; i < ITERS; i++) {
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

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

    // Copy result back to host
    half *h_C_bf16 = new half[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __half2float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f, total_error = 0.0f, total_ref = 0.0f, total_ours=0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if( error > 0.20 ) { // large because of fp4 vs fp32 numerics
            if(error_count < 100) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 700) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        // if (error > max_error) // printf("Error at row %d col %d: %f != %f (ref)\n", i / N, i % N, h_C[i], h_C_ref[i]);
        max_error = std::max(max_error, error);
        total_ref += h_C_ref[i]*h_C_ref[i];
        total_error += error*error;
        total_ours += h_C[i]*h_C[i];
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_fp4;
    delete[] h_B_fp4;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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
    N = 8192;
    run_benchmark(N, N, N);
    // N = 16384;
    // run_benchmark(N, N, N);
    return 0;
}
