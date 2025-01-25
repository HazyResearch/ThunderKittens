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
    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb, Kb>;
    using d_tile = st_bf<Mb, Nb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

__global__  __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();

    constexpr int PIPE_DEPTH = 3;

    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb, Kb>;
    using d_tile = st_bf<Mb, Nb>;
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_DEPTH] = al.allocate<b_tile, PIPE_DEPTH>();
    d_tile (*d_smem) = reinterpret_cast<d_tile*>(&a_smem[0][0]);

    auto all_tmem = allocate_tmem();
    using d_tmem_t = tmem<float, Mb, Nb>;

    d_tmem_t d_tmem = all_tmem.subtile<d_tmem_t>(0, warpgroupid*Nb);

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], mma_done[NUM_CONSUMERS][2];
    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1); 
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); 
        }
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(mma_done[i][0], 0, 1); 
            init_semaphore(mma_done[i][1], 0, 1); 
        }
    }

    int row_idx = warpgroupid == NUM_CONSUMERS ? blockIdx.x*2 : blockIdx.x*2 + warpgroupid;
    int col_idx = blockIdx.y;

    __syncthreads(); 
    
    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::decrease_registers<32>();  

        if(warpid == NUM_WORKERS-4) {
            for (auto idx = 0; idx < g.a.cols / Kb; idx++) {
                tma::expect(inputs_arrived[idx%PIPE_DEPTH], a_smem[0][0], a_smem[0][1], b_smem[0]);
                tma::load_async(a_smem[idx%PIPE_DEPTH][0], g.a, {row_idx,      idx}, inputs_arrived[idx%PIPE_DEPTH]);
                tma::load_async(a_smem[idx%PIPE_DEPTH][1], g.a, {(row_idx+1),  idx}, inputs_arrived[idx%PIPE_DEPTH]);
                tma::load_async(b_smem[idx%PIPE_DEPTH],    g.b, {col_idx,      idx}, inputs_arrived[idx%PIPE_DEPTH]);
                if(idx >= PIPE_DEPTH-1) {
                    wait(inputs_finished[(idx-PIPE_DEPTH+1)%PIPE_DEPTH], ((idx-PIPE_DEPTH+1)/PIPE_DEPTH)%2);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();

        wait(inputs_arrived[0], 0);
        if(warpgroup::warpid() == 0){
            mm_ABt(d_tmem, a_smem[0][warpgroupid], b_smem[0], mma_done[warpgroupid][0]);
        }

        int idx;
        for (idx = 1; idx < g.a.cols / Kb; idx++) {
            wait(inputs_arrived[idx%PIPE_DEPTH], (idx/PIPE_DEPTH)%2);
            if(warpgroup::warpid() == 0) {
                mma_ABt(d_tmem, a_smem[idx%PIPE_DEPTH][warpgroupid], b_smem[idx%PIPE_DEPTH], mma_done[warpgroupid][idx%2]);
            }
            wait(mma_done[warpgroupid][((idx-1)%2)], ((idx-1)/2)%2);
            warpgroup::sync(warpgroupid);
            if(warpgroup::laneid() == 0) arrive(inputs_finished[(idx-1)%PIPE_DEPTH], 1);
        }
        wait(mma_done[warpgroupid][((idx-1)%2)], ((idx-1)/2)%2);
        warpgroup::sync(warpgroupid);
        if(warpgroup::laneid() == 0) arrive(inputs_finished[(idx-1)%PIPE_DEPTH], 1);

        rt_fl<Mb/4, Nb> d_reg;
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
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

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

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(M / (2*Mb), N / Nb, 1);
    dim3 block(NUM_THREADS);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        inner_run(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
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
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

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

    return 0;
}

int main() {
    // int Cblocks = 22, Rblocks = 24;
    // int Cblocks192 = 20, Rblocks192 = 16;
    // run_benchmark<matmul_template<4>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<8>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<12>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    int N;
    N = 8192;
    run_benchmark(N, N, N);
    // N = 3072;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 4096;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 6144;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 8192;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 12288;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 16384;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<2,4,12>>(N, N, N);
    // run_benchmark<matmul_template<3,3,12>>(192*12, 192*11, 8192);
    // run_benchmark<matmul_template<2,4,11>>(128*22, 256* 6, 8192);
    // run_benchmark<matmul_template<2,4,1>>(128 * 132, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(128 * 133, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,8>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,12>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,128>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 16384);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*12*2, 192*11*2, 8192*2);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192*2);
    return 0;
}