#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, -1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int batch; int2 coord; };
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int B, int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : B*M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
      // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64);
        int Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;

        int blocks_per_batch = Rblocks*Cblocks;

        int global_block_id = args.task_iter*gridDim.x + blockIdx.x;
        int batch_idx = global_block_id / blocks_per_batch;
        int block_idx_in_batch = global_block_id % blocks_per_batch;
        if (batch_idx >= args.globals.A.batch) {
            args.num_iters = -1;
            return;
        }
        args.common.batch = batch_idx;
        if (block_idx_in_batch < super_rows * Cblocks) { // 32*16 = 512
            int x = SUPER_M*(block_idx_in_batch/super_repeat) + block_idx_in_batch%SUPER_M;
            args.common.coord = { x, (block_idx_in_batch%super_repeat)/SUPER_M };
        }
        else if (block_idx_in_batch < Rblocks*Cblocks) { // 512
            int remainder_id = block_idx_in_batch - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols/64;  // 64
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.batch, 0, args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.batch, 0, args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            for (int n = 0; n < N_BLOCK; n++) 
                zero(args.state.accum[n]);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::mma_ABt(
                    args.state.accum[n],
                    args.input.a[warpgroup::groupid()],
                    args.input.b[n]
                );
            }
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                   {args.common.batch, 0, args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait();
                }
            }

            // Zero the accumulators
            for(int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n]);
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int B, int M, int N, int K) {
    #pragma omp parallel for collapse(3) // otherwise the CPU version takes for everrrrrr
    for (int b_idx = 0; b_idx < B; b_idx++) { // b
        for (int m_idx = 0; m_idx < M; m_idx++) { // i
            for (int n_idx = 0; n_idx < N; n_idx++) { // j
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < K; k_idx++) { // k
                    sum += a[b_idx * M * K + m_idx * K + k_idx] * b[b_idx * K * N + k_idx * N + n_idx];
                }
                c[b_idx * M * N + m_idx * N + n_idx] = sum;
            }
        }
    }
}

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t B, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    // printf("M: %d, N: %d, K: %d\n", M, N, K);
    global_layout Ag{d_A, B, nullptr, M, K};
    global_layout Bg{d_B, B, nullptr, K, N};
    global_layout Cg{d_C, B, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
int run_benchmark(size_t B, size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  B=" << B << " M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Allocate host memory
    float *h_A = new float[B * M * K];
    float *h_B = new float[B * K * N];
    float *h_C = new float[B * M * N];
    float *h_C_ref = new float[B * M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < B * M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < B * K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, B, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, B*M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, B*K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, B*M*N*sizeof(__nv_bfloat16));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[B * M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[B * K * N];
    for (int i = 0; i < B * M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < B * K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, B*M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, B*K*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(B, M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, B, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, B, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * B * M * N * K; // 2 FLOPs per multiply-add
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
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[B * M * N];
    cudaMemcpy(h_C_bf16, d_C, B*M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < B * M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < B * M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            int b = i / (M * N), row = i % (M * N) / N, col = i % N;
            if(error_count < 20) std::cout << "Error at batch " << b << " row " << row << " col " << col << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Total elements: " << M*N << std::endl;
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
    int N;
    N = 4096;
    run_benchmark<matmul_template<2,4,8>>(2, N, N, N);
    return 0;
}
