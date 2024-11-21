#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
template<int N> __device__ static inline void init_bias(rt_fl<1,N> &acc, const sv_bf<N> &bias) {
    #pragma unroll
    for(int i = 0; i < N; i++) {
        float2 tmp1 = __bfloat1622float2(*(bf16_2*)&bias.data[16*i + 0 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[0].x = tmp1.x;
        acc.tiles[0][i].data[0].y = tmp1.y;
        acc.tiles[0][i].data[1].x = tmp1.x;
        acc.tiles[0][i].data[1].y = tmp1.y;
        float2 tmp2 = __bfloat1622float2(*(bf16_2*)&bias.data[16*i + 8 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[2].x = tmp2.x;
        acc.tiles[0][i].data[2].y = tmp2.y;
        acc.tiles[0][i].data[3].x = tmp2.x;
        acc.tiles[0][i].data[3].y = tmp2.y;
    }
}
__device__ static inline float epilogue(float f) {
    // linear
    // return f;
    // approximate GeLU
    // return f * 0.5f * (1.0f + tanh(f * 0.79788456f * (1 + f * f *0.044715f)));
    // ReLU
    return f > 0.0f ? f : 0.0f;
}
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct linear_template {
    using w_tile = st_bf<4,BLOCK_K/16>;
    using x_tile = st_bf<BLOCK_K/16,BLOCK_N/16>;
    using y_tile = st_bf<4,BLOCK_N/16>;
    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
    using w_global = kittens::gl<bf16, 1, 1, -1, -1, w_tile>;
    using x_global = kittens::gl<bf16, 1, 1, -1, -1, x_tile>;
    using b_global = kittens::gl<bf16, 1, 1,  1, -1>; // just a vector, no need for tma
    using y_global = kittens::gl<bf16, 1, 1, -1, -1, y_tile>;
    struct globals {
        w_global Wg;
        x_global Xg;
        b_global Bg;
        y_global Yg;
    };
    struct input_block { // the chunk of data that the producer and consumer are working on
        w_tile w[NUM_CONSUMER_WARPGROUPS];
        x_tile x;
    };
    static constexpr int INPUT_PIPE_STAGES = 225000/sizeof(input_block), OUTPUT_PIPE_STAGES = 0; // irrelevant for this kernel
    struct output_block {}; // nothing here, we store at the end in the consumer finish
    struct scratch_block { sv_bf<y_tile::width> bias; };
    struct finish_block {
        y_tile y[NUM_CONSUMER_WARPGROUPS];
    };
    struct producer {
        struct state { int row_idx, col_idx, n_blocks; }; // persistent registers
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<24>(); // decrease registers for the producer warpgroup
            s.row_idx = blockIdx.x * NUM_CONSUMER_WARPGROUPS; // tiles vertical per block
            s.col_idx = blockIdx.y; // just 1 tile horizontal per block
            s.n_blocks = g.Wg.cols / w_tile::cols; // number of blocks to process
        }
        __device__ static bool load(state &s, input_block &b, globals &g, semaphore &inputs_arrived, int iter) { // semaphore for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(inputs_arrived, size_bytes<w_tile>*NUM_CONSUMER_WARPGROUPS + size_bytes<x_tile>);
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(b.w[i], g.Wg, {s.row_idx+i, iter}, inputs_arrived);
                }
                tma::load_async(b.x, g.Xg, {iter, s.col_idx}, inputs_arrived);
            }
            else arrive(inputs_arrived);
            return iter < s.n_blocks-1; // return true if there are more blocks to process
        }
    };
    struct consumer {
        struct state { rt_fl<1,y_tile::width> acc; int n_blocks; };
        __device__ static void setup(state &s, scratch_block &scratch, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<480/NUM_CONSUMER_WARPGROUPS - 8*(NUM_CONSUMER_WARPGROUPS>3)>();
            group<NUM_CONSUMER_WARPS>::load(scratch.bias, g.Bg, {blockIdx.y});
            group<NUM_CONSUMER_WARPS>::sync(0);
            init_bias<y_tile::width>(s.acc, scratch.bias);
            s.n_blocks = g.Wg.cols / w_tile::cols;
        }
        __device__ static bool work(state &s, input_block &b, scratch_block &scratch, output_block &o, semaphore &inputs_finished, semaphore &outputs_arrived, int iter) {
            warpgroup::mma_AB(s.acc, b.w[warpgroup::groupid()], b.x);
            warpgroup::mma_async_wait();
            arrive(inputs_finished);
            return iter < s.n_blocks-1;
        }
        __device__ static void finish(state &s, finish_block &f, scratch_block &scratch, globals &g, int _) {
            #pragma unroll
            for(int i = 0; i < y_tile::width; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    s.acc.tiles[0][i].data[j].x = epilogue(s.acc.tiles[0][i].data[j].x);
                    s.acc.tiles[0][i].data[j].y = epilogue(s.acc.tiles[0][i].data[j].y);
                }
            }
            warpgroup::store(f.y[warpgroup::groupid()], s.acc);
            warpgroup::sync();
            if(warpgroup::warpid() == 0) {
                tma::store_async(g.Yg, f.y[warpgroup::groupid()], {blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), blockIdx.y});
            }
            tma::store_async_read_wait();
        }
    };
};

#include <iostream>
#include <random>
#include <cuda_bf16.h>

#include <omp.h>
void cpu_gemm(float* a, float* b, float *bias, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum + bias[j];
            if(c[i * N + j] < 0.0f) c[i * N + j] = 0.0f;
        }
    }
}

int main() {
    // const int M = 3072, N = 12288, K = 3072; using lt = linear_template<192, 192, 64>; // 760 TFLOPs
    const int M = 3072, N = 3072, K = 12288; using lt = linear_template<192, 192, 64>; // 813.5 TFLOPs
    // const int M = 256, N = 12288, K = 3072; using lt = linear_template<128, 192, 64>; // 574.5 TFLOPs
    // const int M = 256, N = 3072, K = 12288; using lt = linear_template<128, 64, 128>; // 433 TFLOPs
    // const int M = 3072, N = 3072, K = 3072; using lt = linear_template<192, 192, 64>; // 740 TFLOPs

    // Allocate host memory
    float *h_W = new float[M * K];
    float *h_X = new float[K * N];
    float *h_b = new float[N];
    float *h_Y = new float[M * N];
    float *h_Y_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_W[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_X[i] = dis(gen);
    for (int i = 0; i < N; ++i) h_b[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    cpu_gemm(h_W, h_X, h_b, h_Y_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_W, *d_X, *d_b, *d_Y;
    cudaMalloc(&d_W, M*K*2);
    cudaMalloc(&d_X, K*N*2);
    cudaMalloc(&d_b,   N*2);
    cudaMalloc(&d_Y, M*N*2);

    std::cout << "Allocated device memory" << std::endl;

    std::cout << "w_tile::rows=" << lt::w_tile::rows << " w_tile::cols=" << lt::w_tile::cols << std::endl;
    std::cout << "x_tile::rows=" << lt::x_tile::rows << " x_tile::cols=" << lt::x_tile::cols << std::endl;
    std::cout << "y_tile::rows=" << lt::y_tile::rows << " y_tile::cols=" << lt::y_tile::cols << std::endl;
    lt::w_global Wg{d_W, nullptr, nullptr, M, K};
    lt::x_global Xg{d_X, nullptr, nullptr, K, N};
    lt::b_global Bg{d_b, nullptr, nullptr, nullptr, N};
    lt::y_global Yg{d_Y, nullptr, nullptr, M, N};
    lt::globals globals{Wg, Xg, Bg, Yg};

    std::cout << "Allocated memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_W_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_X_bf16 = new __nv_bfloat16[K * N];
    __nv_bfloat16 *h_b_bf16 = new __nv_bfloat16[N];
    for (int i = 0; i < M * K; ++i) h_W_bf16[i] = __float2bfloat16(h_W[i]);
    for (int i = 0; i < K * N; ++i) h_X_bf16[i] = __float2bfloat16(h_X[i]);
    for (int i = 0; i < N; ++i) h_b_bf16[i] = __float2bfloat16(h_b[i]);

    cudaMemcpy(d_W, h_W_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X_bf16, K*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_bf16, N*2, cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = 226000; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(prototype::pc<lt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // Launch kernel
    dim3 grid(M / (lt::y_tile::rows*prototype::num_consumer_warpgroups<lt>), N / lt::y_tile::cols); // rows, cols
    dim3 block(prototype::num_threads<lt>);

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/lt::w_tile::cols << " reduction block dimension\n";
    std::cout << "Kernel has " << lt::INPUT_PIPE_STAGES << " input pipeline stages and " << lt::OUTPUT_PIPE_STAGES << " output pipeline stages\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 100;
    for(int i = 0; i < ITERS; i++) {
        prototype::pc<lt><<<grid, block, mem_size>>>(globals);
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
    __nv_bfloat16 *h_Y_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_Y_bf16, d_Y, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_Y[i] = __bfloat162float(h_Y_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_Y[i] - h_Y_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_Y[i] << " != " << h_Y_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_W;
    delete[] h_X;
    delete[] h_b;
    delete[] h_Y;
    delete[] h_Y_ref;
    delete[] h_W_bf16;
    delete[] h_X_bf16;
    delete[] h_b_bf16;
    delete[] h_Y_bf16;
    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_b);
    cudaFree(d_Y);

    return 0;
}
