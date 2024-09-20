#include "kittens.cuh"
#include "prototype.cuh"
using namespace kittens;
using namespace kittens::prototype;

struct prep_layout {
    using tile = st_bf<64,64>;
    using x_layout = gl<bf16, 1, 1, -1, -1, tile>;
    using y_layout = gl<bf16, 1, 5, -1, -1, tile>; // 5 stack
    struct input_block  { tile x[2][2]; };
    struct output_block { tile y[5]; };
    struct globals {
        x_layout x;
        y_layout y;
    };
};
struct prep_ker {
    using layout = prep_layout;
    static constexpr int NUM_CONSUMER_WARPS=4, OUTPUT_PIPE_STAGES=2, NUM_BLOCKS=1;
    __device__ static inline int iters(typename layout::globals &g) {
        int iters = ((g.x.cols / 128)*(g.x.rows / 128) - (blockIdx.x+1)) / 132; // use all 132 SM's
        return iters;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
            warpgroup::producer_registers(); // decrease registers for the producer warpgroup
        }
        __device__ static void load(producer_load_args<layout> args) { // barrier for the producer to load into
            int global_block_idx = (args.iter*132+blockIdx.x), num_block_cols = (args.globals.x.cols / layout::tile::cols) / 2;
            kittens::index idx = {global_block_idx / num_block_cols, global_block_idx % num_block_cols};
            int row_offset_blocks = (args.globals.x.rows / layout::tile::rows) / 2, col_offset_blocks = (args.globals.x.cols / layout::tile::cols) / 2;
            if(warpgroup::warpid() == args.iter%4) {
                tma::expect_bytes(args.inputs_arrived, sizeof(layout::input_block));
                tma::load_async(args.input.x[0][0], args.globals.x, {idx.r, idx.c}, args.inputs_arrived);
                tma::load_async(args.input.x[0][1], args.globals.x, {idx.r, idx.c+col_offset_blocks}, args.inputs_arrived);
                tma::load_async(args.input.x[1][0], args.globals.x, {idx.r+row_offset_blocks, idx.c}, args.inputs_arrived);
                tma::load_async(args.input.x[1][1], args.globals.x, {idx.r+row_offset_blocks, idx.c+col_offset_blocks}, args.inputs_arrived);
                arrive(args.inputs_arrived, 3);
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int global_block_idx = (args.iter*132+blockIdx.x), num_block_cols = (args.globals.y.cols / layout::tile::cols);
            kittens::index idx = {global_block_idx / num_block_cols, global_block_idx % num_block_cols};
            if(warpgroup::warpid() == args.iter%4) {
                for(int i = 0; i < 5; i++)
                    tma::store_async(args.globals.y, args.output.y[i], {i, idx.r, idx.c});
                tma::store_async_read_wait();
                arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
            warpgroup::increase_registers<256>();
        }
        __device__ static void work(consumer_work_args<layout> args) {
            rt_fl<16,64> x[2][2];
            rt_fl<16,64> y[5];
            warpgroup::load(x[0][0], args.input.x[0][0]);
            warpgroup::load(x[0][1], args.input.x[0][1]);
            warpgroup::load(x[1][0], args.input.x[1][0]);
            warpgroup::load(x[1][1], args.input.x[1][1]);
            arrive(args.inputs_finished);
            add(y[0], x[0][0], x[1][1]); // x11 + x22
            add(y[1], x[1][0], x[1][1]); // x21 + x22
            add(y[2], x[0][0], x[0][1]); // x11 + x12
            sub(y[3], y[1], y[0]); // x21 - x11
            sub(y[4], y[2], y[0]); // x12 - x22
            warpgroup::store(args.output.y[0], y[0]);
            warpgroup::store(args.output.y[1], y[1]);
            warpgroup::store(args.output.y[2], y[2]);
            warpgroup::store(args.output.y[3], y[3]);
            warpgroup::store(args.output.y[4], y[4]);
            warpgroup::sync();
            arrive(args.outputs_arrived);
        }
    };
};

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct matmul_layout {
    using a_tile = st_bf<64, BLOCK_K>;
    using b_tile = st_bf<BLOCK_K, BLOCK_N>;
    using c_tile = st_bf<64, BLOCK_N>;
    struct globals { // global layout (here with TMA descriptors)
        gl<bf16, 1, 1, -1, -1, a_tile> a;
        gl<bf16, 1, 1, -1, -1, b_tile> b;
        gl<bf16, 1, 1, -1, -1, c_tile> c;
    };
    struct input_block {
        a_tile a[BLOCK_M/64];
        b_tile b;
    };
    struct consumer_state { rt_fl<16, BLOCK_N> accumulator;   };
    struct finish_block   { c_tile c[BLOCK_M/64]; };
};
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct matmul_template {
    using layout = matmul_layout<BLOCK_M, BLOCK_N, BLOCK_K>;
    template<int C_00, int C_01, int C_10, int C_11>
    struct ker {
        static_assert(C_00 >= -1 && C_00 <= 1 && C_01 >= -1 && C_01 <= 1 && C_10 >= -1 && C_10 <= 1 && C_11 >= -1 && C_11 <= 1, "C_00, C_01, C_10, C_11 must be between -1 and 1");
        using layout = layout;
        static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
        __device__ static inline int iters(typename layout::globals &g) { return g.a.cols / BLOCK_K; }
        struct producer {
            __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
                warpgroup::producer_registers(); // decrease registers for the producer warpgroup
            }
            __device__ static void load(producer_load_args<layout> args) { // barrier for the producer to load into
                if(warpgroup::warpid() == 0) {
                    tma::expect_bytes(args.inputs_arrived, sizeof(layout::input_block));
                    for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++)
                        tma::load_async(args.input.a[i], args.globals.a, {(int)blockIdx.x*NUM_CONSUMER_WARPGROUPS+i, args.iter}, args.inputs_arrived);
                    tma::load_async(args.input.b, args.globals.b, {args.iter, (int)blockIdx.y}, args.inputs_arrived);
                }
                else arrive(args.inputs_arrived);
            }
        };
        struct consumer {
            __device__ static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
                warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
                zero(args.state.accumulator);
            }
            __device__ static void work(consumer_work_args<layout> args) {
                warpgroup::mma_AB(args.state.accumulator, args.input.a[warpgroup::groupid()], args.input.b);
                warpgroup::mma_async_wait();
                arrive(args.inputs_finished);
            }
            __device__ static void finish(consumer_finish_args<layout> args) {
                int warpgroup_row_offset = blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), warpgroup_col_offset = blockIdx.y;
                int row_offset_blocks = (args.globals.c.rows / layout::c_tile::rows) / 2, col_offset_blocks = (args.globals.c.cols / layout::c_tile::cols) / 2;
                warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accumulator);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    if constexpr (C_00 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset, warpgroup_col_offset});
                    }
                    else if constexpr (C_01 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset, warpgroup_col_offset + col_offset_blocks});
                    }
                    else if constexpr (C_10 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset + row_offset_blocks, warpgroup_col_offset});
                    }
                    else if constexpr (C_11 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset + row_offset_blocks, warpgroup_col_offset + col_offset_blocks});
                    }
                }
                mul(args.state.accumulator, args.state.accumulator, -1.f);
                tma::store_async_read_wait(); // make sure the store is finished before we start writing back
                warpgroup::sync();
                warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accumulator);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    if constexpr (C_00 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset, warpgroup_col_offset});
                    }
                    else if constexpr (C_01 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset, warpgroup_col_offset + col_offset_blocks});
                    }
                    else if constexpr (C_10 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset + row_offset_blocks, warpgroup_col_offset});
                    }
                    else if constexpr (C_11 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {warpgroup_row_offset + row_offset_blocks, warpgroup_col_offset + col_offset_blocks});
                    }
                }
            }
        };
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
    const int BIG_M = 16384, BIG_N = 16384, BIG_K = 16384;
    const int M = BIG_M / 2, N = BIG_N / 2, K = BIG_K / 2; using mmt = matmul_template<128, 256, 64>; // 813.5 TFLOPs

    using a_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().a[0])>::type;
    using b_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().b)>::type;
    using c_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::finish_block>().c[0])>::type;
    using a_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().a)>::type;
    using b_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().b)>::type;
    using c_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().c)>::type;
    using globals  = typename mmt::layout::globals;

    std::cout << "Has store: "  << (bool)kittens::prototype::detail::has_store<mmt> << '\n';
    std::cout << "Has finish: " << (bool)kittens::prototype::detail::has_finish<mmt> << '\n';

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
    // cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    kittens::bf16 *d_prep_X, *d_prep_Y;
    cudaMalloc(&d_prep_X, BIG_M*BIG_K*2);
    cudaMalloc(&d_prep_Y, M*K*10);

    std::cout << "Allocated prep memory" << std::endl;
    std::cout << "Timing prep kernel:" << std::endl;
    using prep_ker = prep_ker;
    prep_ker::layout::x_layout Xg(d_prep_X, nullptr, nullptr, BIG_M, BIG_K);
    prep_ker::layout::y_layout Yg(d_prep_Y, nullptr, nullptr, M, K);
    prep_ker::layout::globals prep_G{Xg, Yg};

    unsigned long prep_mem_size = 113000*2; // need to launch two blocks if possible.
    cudaFuncSetAttribute(prototype::pc<prep_ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem_size);
    dim3 prep_grid(132); // rows, cols
    dim3 prep_block(256);

    for(int i = 0; i < 2; i++) {
        prototype::pc<prep_ker><<<prep_grid, prep_block, prep_mem_size>>>(prep_G);
    }
    cudaDeviceSynchronize();

    std::cout << "Finished warmup!" << std::endl;
    const int PREP_ITERS = 10;
    auto prep_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < PREP_ITERS; i++) {
        prototype::pc<prep_ker><<<prep_grid, prep_block, prep_mem_size>>>(prep_G);
    }
    cudaDeviceSynchronize();
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prep_diff = prep_end - prep_start;
    double prep_seconds = prep_diff.count();
    std::cout << "Prep kernel execution time (per iter): " << prep_seconds*1e6 / PREP_ITERS << " us\n";
    // Check for CUDA errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    Xg.cleanup();
    Yg.cleanup();
    cudaFree(d_prep_X);
    cudaFree(d_prep_Y);

    // // Allocate device memory
    // __nv_bfloat16 *d_A, *d_B, *d_C;
    // cudaMalloc(&d_A, M*K*2);
    // cudaMalloc(&d_B, K*N*2);
    // cudaMalloc(&d_C, BIG_M*BIG_N*2);

    // std::cout << "Allocated device memory" << std::endl;

    // std::cout << "a_tile::rows=" << a_tile::rows << " a_tile::cols=" << a_tile::cols << std::endl;
    // std::cout << "b_tile::rows=" << b_tile::rows << " b_tile::cols=" << b_tile::cols << std::endl;
    // std::cout << "c_tile::rows=" << c_tile::rows << " c_tile::cols=" << c_tile::cols << std::endl;
    // a_global Ag{d_A, nullptr, nullptr, M, K};
    // b_global Bg{d_B, nullptr, nullptr, K, N};
    // c_global Cg{d_C, nullptr, nullptr, BIG_M, BIG_N};
    // globals G{Ag, Bg, Cg};

    // std::cout << "Allocated memory" << std::endl;

    // // Convert to __nv_bfloat16 and copy to device
    // __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    // __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    // for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    // for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    // cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

    // std::cout << "Copied matrices to device" << std::endl;

    // unsigned long mem_size = MAX_SHARED_MEMORY; // need to launch two blocks if possible.

    // using ker = mmt::template ker<1, 0, 0, 0>;
    
    // cudaFuncSetAttribute(prototype::pc<ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // // Launch kernel
    // dim3 grid(M / (c_tile::rows*prototype::num_consumer_warpgroups<ker>), N / c_tile::cols); // rows, cols
    // dim3 block(prototype::num_threads<ker>);

    // // Start timing
    // cudaDeviceSynchronize();
    // std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/a_tile::cols << " reduction block dimension\n";
    // std::cout << "Kernel has " << kittens::prototype::input_pipe_stages<mmt> << " input pipeline stages and " << kittens::prototype::output_pipe_stages<mmt> << " output pipeline stages\n";

    // constexpr int ITERS = 1;
    // for(int i = 0; i < 5; i++) {
    //     prototype::pc<ker><<<grid, block, mem_size>>>(G);
    // }
    // cudaDeviceSynchronize();

    // std::cout << "Finished warmup!" << std::endl;
    // auto start = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < ITERS*7; i++) {
    //     prototype::pc<ker><<<grid, block, mem_size>>>(G);
    // }
    // cudaDeviceSynchronize();
    // // End timing
    // auto end = std::chrono::high_resolution_clock::now();

    // // Calculate duration
    // std::chrono::duration<double> diff = end - start;
    // double seconds = diff.count();

    // // Calculate TFLOPs
    // double flops = double(2.0) * BIG_M * BIG_N * BIG_K * ITERS; // 2 FLOPs per multiply-add
    // double expected_tax = (double(BIG_M+BIG_N)*BIG_K*2 + 5*((M+N)*K)) / 3e12;
    // double tflops = (flops / seconds) / 1e12;
    // double taxed_tflops = (flops / (expected_tax+seconds)) / 1e12;

    // std::cout << "Kernel execution time: " << seconds*1e6 << " us\n";
    // std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    // std::cout << "Expected pre-cost: " << expected_tax*1e6 << " us\n";
    // std::cout << "Taxed performance: " << taxed_tflops << " TFLOPs\n";
    // // Check for CUDA errors
    // cudaError_t cudaStatus = cudaGetLastError();
    // if (cudaStatus != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    //     // Optionally, you might want to exit the program or handle the error in some way
    //     return -1;
    // }

    // // Copy result back to host
    // __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    // cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    // std::cout << "Copied result back to host" << std::endl;

    // // Convert result back to float for comparison
    // for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    // std::cout << "Converted result back to float" << std::endl;

    // Check result
    // float max_error = 0.0f;
    // int error_count = 0;
    // for (int i = 0; i < M * N; ++i) {
    //     float error = std::abs(h_C[i] - h_C_ref[i]);
    //     if(error > 1.0) { // large because of bf16 vs fp32 numerics
    //         if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
    //         else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
    //         error_count++;
    //     }
    //     max_error = std::max(max_error, error);
    // }

    // std::cout << "Max error: " << max_error << std::endl;
    // std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;
    // delete[] h_C_ref;
    // delete[] h_A_bf16;
    // delete[] h_B_bf16;
    // delete[] h_C_bf16;
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);

    return 0;
}
