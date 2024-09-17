#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct matmul_template {
    using a_tile = st_bf<4,BLOCK_K/16>;
    using b_tile = st_bf<BLOCK_K/16,BLOCK_N/16>;
    using c_tile = st_bf<4,BLOCK_N/16>;
    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
    using a_global = kittens::gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_global = kittens::gl<bf16, 1, 1, -1, -1, b_tile>;
    using c_global = kittens::gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals {
        a_global Ag;
        b_global Bg;
        c_global Cg;
    };
    struct input_block { // the chunk of data that the producer and consumer are working on
        a_tile a_block[NUM_CONSUMER_WARPGROUPS];
        b_tile b_block;
    };
    static constexpr int INPUT_PIPE_STAGES = 220000/sizeof(input_block), OUTPUT_PIPE_STAGES = 0; // irrelevant for this kernel
    struct output_block {}; // nothing here, we store at the end in the consumer finish
    struct scratch_block {};
    struct finish_block {
        c_tile c_block[NUM_CONSUMER_WARPGROUPS];
    };
    struct producer {
        struct state { int row_idx, col_idx, n_blocks; }; // persistent registers
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<24>(); // decrease registers for the producer warpgroup
            s.row_idx = blockIdx.x * NUM_CONSUMER_WARPGROUPS; // tiles vertical per block
            s.col_idx = blockIdx.y; // just 1 tile horizontal per block
            s.n_blocks = g.Ag.cols / a_tile::cols; // number of blocks to process
        }
        __device__ static bool load(state &s, input_block &b, globals &g, barrier &inputs_arrived, int iter) { // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(inputs_arrived, size_bytes<a_tile>*NUM_CONSUMER_WARPGROUPS + size_bytes<b_tile>);
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(b.a_block[i], g.Ag, {s.row_idx+i, iter}, inputs_arrived);
                }
                tma::load_async(b.b_block, g.Bg, {iter, s.col_idx}, inputs_arrived);
            }
            else arrive(inputs_arrived);
            return iter < s.n_blocks-1; // return true if there are more blocks to process
        }
    };
    struct consumer {
        struct state { rt_fl<1,c_tile::width> acc; int n_blocks; };
        __device__ static void setup(state &s, scratch_block &_, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<480/NUM_CONSUMER_WARPGROUPS - 8*(NUM_CONSUMER_WARPGROUPS>3)>();
            zero(s.acc);
            s.n_blocks = g.Ag.cols / a_tile::cols;
        }
        __device__ static bool work(state &s, input_block &b, scratch_block &_, output_block &o, barrier &inputs_finished, barrier &outputs_arrived, int iter) {
            warpgroup::mma_AB(s.acc, b.a_block[warpgroup::groupid()], b.b_block);
            warpgroup::mma_async_wait();
            arrive(outputs_arrived); // we have no outputs, so we can do this early. (they're always ready.)
            arrive(inputs_finished);
            return iter < s.n_blocks-1;
        }
        __device__ static void finish(state &s, finish_block &f, scratch_block &scratch, globals &g, int _) {
            warpgroup::store(f.c_block[warpgroup::groupid()], s.acc);
            warpgroup::sync();
            if(warpgroup::warpid() == 0) {
                tma::store_async(g.Cg, f.c_block[warpgroup::groupid()], {blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), blockIdx.y});
            }
            tma::store_async_read_wait();
        }
    };
};

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
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    // const int M = 3072, N = 12288, K = 3072; using mmt = matmul_template<192, 192, 64>; // 760 TFLOPs
    // const int M = 3072, N = 3072, K = 12288; using mmt = matmul_template<192, 192, 64>; // 813.5 TFLOPs
    // const int M = 256, N = 12288, K = 3072; using mmt = matmul_template<128, 192, 64>; // 574.5 TFLOPs
    // const int M = 256, N = 3072, K = 12288; using mmt = matmul_template<128, 64, 128>; // 433 TFLOPs
    const int M = 3072, N = 3072, K = 3072; using mmt = matmul_template<192, 192, 64>; // 740 TFLOPs

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*2);
    cudaMalloc(&d_B, K*N*2);
    cudaMalloc(&d_C, M*N*2);

    std::cout << "Allocated device memory" << std::endl;

    std::cout << "a_tile::rows=" << mmt::a_tile::rows << " a_tile::cols=" << mmt::a_tile::cols << std::endl;
    std::cout << "b_tile::rows=" << mmt::b_tile::rows << " b_tile::cols=" << mmt::b_tile::cols << std::endl;
    std::cout << "c_tile::rows=" << mmt::c_tile::rows << " c_tile::cols=" << mmt::c_tile::cols << std::endl;
    mmt::a_global Ag{d_A, nullptr, nullptr, M, K};
    mmt::b_global Bg{d_B, nullptr, nullptr, K, N};
    mmt::c_global Cg{d_C, nullptr, nullptr, M, N};
    mmt::globals globals{Ag, Bg, Cg};

    std::cout << "Allocated memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = 226000; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(prototype::pc<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // Launch kernel
    dim3 grid(M / (mmt::c_tile::rows*prototype::num_consumer_warpgroups<mmt>), N / mmt::c_tile::cols); // rows, cols
    dim3 block(prototype::num_threads<mmt>);

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/mmt::a_tile::cols << " reduction block dimension\n";
    std::cout << "Kernel has " << mmt::INPUT_PIPE_STAGES << " input pipeline stages and " << mmt::OUTPUT_PIPE_STAGES << " output pipeline stages\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 100;
    for(int i = 0; i < ITERS; i++) {
        prototype::pc<mmt><<<grid, block, mem_size>>>(globals);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K * ITERS; // 2 FLOPs per multiply-add
    double tflops = (flops / seconds) / 1e12;

    std::cout << "Kernel execution time: " << seconds << " seconds\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaError_t cudaStatus = cudaGetLastError();
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
