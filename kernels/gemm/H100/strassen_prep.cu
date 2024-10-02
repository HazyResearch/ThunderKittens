#include "kittens.cuh"
#include "prototype.cuh"
using namespace kittens;
using namespace kittens::prototype;

template<int VEC_LENGTH> struct prep_layout {
    using vec = sv_hf<VEC_LENGTH>;
    using x_layout = gl<half, 1, 1, -1, -1, vec>;
    using y_layout = gl<half, 1, 5, -1, -1, vec>; // 5 stack
    struct input_block  { vec x[2][2]; };
    struct output_block { vec y[5]; };
    struct globals {
        x_layout x;
        y_layout y;
    };
};
template<int VEC_LENGTH=1024> struct prep_ker {
    using layout = prep_layout<VEC_LENGTH>;
    static constexpr int NUM_CONSUMER_WARPS=4, OUTPUT_PIPE_STAGES=4, NUM_BLOCKS=1, INPUT_PIPE_STAGES=4, DEBUG=0;
    __device__ static inline bool task_coord(kittens::coord &coords, const typename layout::globals &g, int iter) { return iter < gridDim.x*gridDim.y*gridDim.z; } // go away
    __device__ static inline int iters(const typename layout::globals &g, const kittens::coord &tc) {
        int iters = ((g.y.cols / layout::vec::length)*(g.y.rows / ((1))) - (blockIdx.x+1) + gridDim.x) / gridDim.x;
        return iters;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {}
        __device__ static void load(producer_load_args<layout> args) { // barrier for the producer to load into
            int global_block_idx = (args.iter*gridDim.x+blockIdx.x), num_block_cols = (args.globals.y.cols / layout::vec::length);
            kittens::coord idx = {global_block_idx / num_block_cols, global_block_idx % num_block_cols};
            int row_offset_blocks = (args.globals.y.rows / ((1))), col_offset_blocks = (args.globals.y.cols / layout::vec::length);
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(args.inputs_arrived, sizeof(layout::input_block));
                tma::load_async(args.input.x[0][0], args.globals.x, {idx.r, idx.c}, args.inputs_arrived);
                tma::load_async(args.input.x[0][1], args.globals.x, {idx.r, idx.c+col_offset_blocks}, args.inputs_arrived);
                tma::load_async(args.input.x[1][0], args.globals.x, {idx.r+row_offset_blocks, idx.c}, args.inputs_arrived);
                tma::load_async(args.input.x[1][1], args.globals.x, {idx.r+row_offset_blocks, idx.c+col_offset_blocks}, args.inputs_arrived);
                arrive(args.inputs_arrived, 3);
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int global_block_idx = (args.iter*gridDim.x+blockIdx.x), num_block_cols = (args.globals.y.cols / layout::vec::length);
            kittens::coord idx = {global_block_idx / num_block_cols, global_block_idx % num_block_cols};
            if(warpgroup::warpid() == 1) {
                #pragma unroll
                for(int i = 0; i < 5; i++) {
                    tma::store_async(args.globals.y, args.output.y[i], {i, idx.r, idx.c});
                }
                tma::store_async_read_wait();
                arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {}
        __device__ static void work(consumer_work_args<layout> args) {
            rv_fl<layout::vec::length/4, naive_l> x[2][2];
            rv_fl<layout::vec::length/4, naive_l> y[5];
            warpgroup::load(x[0][0], args.input.x[0][0]);
            warpgroup::load(x[0][1], args.input.x[0][1]);
            warpgroup::load(x[1][0], args.input.x[1][0]);
            warpgroup::load(x[1][1], args.input.x[1][1]);
            __syncwarp();
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
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <omp.h>

void strassen_prep_cpu(const float* A, float* C, int M, int N) {
    int half_M = M / 2;
    int half_N = N / 2;
    int quarter_size = half_M * half_N;

    for (int i = 0; i < quarter_size; i++) {
        int row = i / half_N;
        int col = i % half_N;

        float A11 = A[row * N + col];
        float A12 = A[row * N + col + half_N];
        float A21 = A[(row + half_M) * N + col];
        float A22 = A[(row + half_M) * N + col + half_N];

        // C11+C22
        C[i] = A11 + A22;
        // C21+C22
        C[quarter_size + i] = A21 + A22;
        // C11+C12
        C[2 * quarter_size + i] = A11 + A12;
        // C21-C11
        C[3 * quarter_size + i] = A21 - A11;
        // C12-C22
        C[4 * quarter_size + i] = A12 - A22;
    }
}

// First we need to validate the prep kernel.
template<typename T>
void verify_prep_kernel(int M, int N) {
    // Allocate host memory
    float *h_A_cpu = new float[M * N];
    float *h_C_cpu = new float[5 * (M/2) * (N/2)];
    T *h_A_gpu = new T[M * N];
    T *h_C_gpu = new T[5 * (M/2) * (N/2)];
    float *h_C_gpu_out = new float[5 * (M/2) * (N/2)];

    // Initialize matrix with random values
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    static_assert(std::is_same_v<T, half>);
    for (int i = 0; i < M * N; ++i) {
        h_A_cpu[i] = dis(gen);
        h_A_gpu[i] = __float2half(h_A_cpu[i]);
    }
    // Allocate device memory
    T *d_A, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(T));
    cudaMalloc(&d_C, 5 * (M/2) * (N/2) * sizeof(T));

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A_gpu, M * N * sizeof(T), cudaMemcpyHostToDevice);

    // Call the GPU kernel (assuming it's implemented elsewhere)
    using pk = prep_ker<1024>;
    pk::layout::x_layout Xg(d_A, nullptr, nullptr, M, N);
    pk::layout::y_layout Yg(d_C, nullptr, nullptr, M/2, N/2);
    pk::layout::globals prep_G{Xg, Yg};
    unsigned long prep_mem_size = 100000; // Adjust if needed
    cudaFuncSetAttribute(prototype::pc<pk>, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem_size);
    dim3 prep_grid(132); // Adjust if needed
    dim3 prep_block(256);

    prototype::pc<pk><<<prep_grid, prep_block, prep_mem_size>>>(prep_G);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_C_gpu, d_C, 5 * (M/2) * (N/2) * sizeof(T), cudaMemcpyDeviceToHost);
    // Cast back to float for comparison
    for (int i = 0; i < 5 * (M/2) * (N/2); ++i) {
        h_C_gpu_out[i] = __half2float(h_C_gpu[i]);
    }

    // Check for CUDA errors after kernel execution
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Prep kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Clean up and return or exit
        delete[] h_A_cpu;
        delete[] h_A_gpu;
        delete[] h_C_gpu;
        delete[] h_C_gpu_out;
        cudaFree(d_A);
        cudaFree(d_C);
        return; // or exit(1);
    }

    // Check for any errors during kernel execution
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching prep kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Clean up and return or exit
        delete[] h_A_cpu;
        delete[] h_A_gpu;
        delete[] h_C_gpu;
        delete[] h_C_gpu_out;
        cudaFree(d_A);
        cudaFree(d_C);
        return; // or exit(1);
    }

    // Call the CPU version
    strassen_prep_cpu(h_A_cpu, h_C_cpu, M, N);

    // Compare results
    double max_diff = 0.0;
    int wrong_count = 0;
    for (int i = 0; i < 5 * (M/2) * (N/2); ++i) {
        double diff = std::abs(h_C_gpu_out[i] - h_C_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 0.01 && wrong_count < 512) {
            int row = i / (N/2);
            int col = i % (N/2);
            int block = row / (M/2);
            row %= M/2;
            std::cout << "Wrong element #" << wrong_count+1 << " at (" << block << ", " << row << ", " << col << "): "
                      << "GPU=" << h_C_gpu_out[i] << ", CPU=" << h_C_cpu[i] << ", diff=" << diff << std::endl;
            wrong_count++;
        }
    }

    std::cout << "Max difference: " << max_diff << std::endl;

    // Clean up
    delete[] h_A_cpu;
    delete[] h_A_gpu;
    delete[] h_C_gpu;
    delete[] h_C_gpu_out;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_C);
}

// int main() {
//     verify_prep_kernel<half>(16384, 16384);
// }
