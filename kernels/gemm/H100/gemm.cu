#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct matmul_layout {
    struct globals {
        kittens::gl<bf16, 1, 1, -1, -1, st_bf<64,      BLOCK_K>> a;
        kittens::gl<bf16, 1, 1, -1, -1, st_bf<BLOCK_K, BLOCK_N>> b;
        kittens::gl<bf16, 1, 1, -1, -1, st_bf<64,      BLOCK_N>> c;
    };
    struct input_block {
        st_bf<64,      BLOCK_K> a[BLOCK_M/64];
        st_bf<BLOCK_K, BLOCK_N> b;
    };
    struct producer_state { int row_idx, col_idx, n_blocks; };
    struct consumer_state { rt_fl<16, BLOCK_N> acc; int n_blocks; };
    struct finish_block   { st_bf<64, BLOCK_N> c[BLOCK_M/64]; };
};
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct matmul_template {
    using layout = matmul_layout<BLOCK_M, BLOCK_N, BLOCK_K>;
    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
    static constexpr int INPUT_PIPE_STAGES = 226000/sizeof(typename layout::input_block), OUTPUT_PIPE_STAGES = 0; // irrelevant for this kernel
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
            warpgroup::decrease_registers<24>(); // decrease registers for the producer warpgroup
            args.state.row_idx = blockIdx.x * NUM_CONSUMER_WARPGROUPS; // tiles vertical per block
            args.state.col_idx = blockIdx.y; // just 1 tile horizontal per block
            args.state.n_blocks = args.globals.a.cols / BLOCK_K; // number of blocks to process
        }
        __device__ static bool load(producer_load_args<layout> args) { // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(args.inputs_arrived, size_bytes<st_bf<64, BLOCK_K>> * NUM_CONSUMER_WARPGROUPS + size_bytes<st_bf<BLOCK_K, BLOCK_N>>);
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(args.input.a[i], args.globals.a, {args.state.row_idx+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.b, {args.iter, args.state.col_idx}, args.inputs_arrived);
            }
            else arrive(args.inputs_arrived);
            return args.iter < args.state.n_blocks-1; // return true if there are more blocks to process
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
            warpgroup::increase_registers<480/NUM_CONSUMER_WARPGROUPS - 8*(NUM_CONSUMER_WARPGROUPS>3)>();
            zero(args.state.acc);
            args.state.n_blocks = args.globals.a.cols / BLOCK_K;
        }
        __device__ static bool work(consumer_work_args<layout> args) {
            warpgroup::mma_AB(args.state.acc, args.input.a[warpgroup::groupid()], args.input.b);
            warpgroup::mma_async_wait();
            arrive(args.inputs_finished);
            return args.iter < args.state.n_blocks-1;
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.acc);
            warpgroup::sync();
            if(warpgroup::warpid() == 0) {
                tma::store_async(args.globals.c, args.finish.c[warpgroup::groupid()], {blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), blockIdx.y});
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
    // const int M = 3072, N = 3072, K = 3072; using mmt = matmul_template<192, 192, 64>; // 740 TFLOPs
    const int M = 3072, N = 3072, K = 12288; using mmt = matmul_template<192, 192, 64>; // 813.5 TFLOPs

    using a_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().a[0])>::type;
    using b_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().b)>::type;
    using c_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::finish_block>().c[0])>::type;
    using a_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().a)>::type;
    using b_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().b)>::type;
    using c_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().c)>::type;
    using globals  = typename mmt::layout::globals;

    std::cout << "Has store: "  << kittens::prototype::has_store<mmt> << '\n';
    std::cout << "Has finish: " << kittens::prototype::has_finish<mmt> << '\n';

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

    std::cout << "a_tile::rows=" << a_tile::rows << " a_tile::cols=" << a_tile::cols << std::endl;
    std::cout << "b_tile::rows=" << b_tile::rows << " b_tile::cols=" << b_tile::cols << std::endl;
    std::cout << "c_tile::rows=" << c_tile::rows << " c_tile::cols=" << c_tile::cols << std::endl;
    a_global Ag{d_A, nullptr, nullptr, M, K};
    b_global Bg{d_B, nullptr, nullptr, K, N};
    c_global Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};

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
    dim3 grid(M / (c_tile::rows*prototype::num_consumer_warpgroups<mmt>), N / c_tile::cols); // rows, cols
    dim3 block(prototype::num_threads<mmt>);

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/a_tile::cols << " reduction block dimension\n";
    std::cout << "Kernel has " << mmt::INPUT_PIPE_STAGES << " input pipeline stages and " << mmt::OUTPUT_PIPE_STAGES << " output pipeline stages\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 100;
    for(int i = 0; i < ITERS; i++) {
        prototype::pc<mmt><<<grid, block, mem_size>>>(G);
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
