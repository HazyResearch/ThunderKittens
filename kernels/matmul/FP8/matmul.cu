#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_fl8_e4m3<64, 128>; // SA: note that if we could accum in fp16, then we could use <64, 256>
    using  store_tile     = st_fl8_e4m3<64, 128*N_BLOCK>;
    using  global_layout  = gl<fp8e4m3, 1, 1, -1, -1, base_tile>;
    using  store_layout   = gl<fp8e4m3, 1, 1, -1, -1, store_tile>;
    struct globals        { global_layout A, B; store_layout C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { store_tile c[M_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { 
        rt_fl<16, store_tile::cols> accum;  // Changed to single tall accumulator
        rt_fl8_e4m3<16, store_tile::cols> accum_fp8;  // Changed to match tall format
    };
};
template<int _M_BLOCK=2, int _N_BLOCK=2, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using tall_tile = st_fl8_e4m3<layout::store_tile::cols, 128>;  // Changed to tall tile

    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64), Cblocks = args.globals.C.cols / (N_BLOCK*128);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols/128;
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
                for(int i = 0; i < M_BLOCK; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<tall_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            copy(args.state.accum_fp8, args.state.accum);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum_fp8);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                   {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k]; // mma_ABt
                // sum += a[i * K + k] * b[k * N + j]; // mma_AB
            }
            c[i * N + j] = sum;
        }
    }
}


template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, fp8e4m3 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using store_layout  = typename mmt::layout::store_layout;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    store_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

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
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen) * 0.1f;

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    fp8e4m3 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(fp8e4m3));
    cudaMalloc(&d_B, K*N*sizeof(fp8e4m3));
    cudaMalloc(&d_C, M*N*sizeof(fp8e4m3));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_fp8_e4m3 and copy to device
    __nv_fp8_e4m3 *h_A_fp8 = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8 = new __nv_fp8_e4m3[K * N];
    for (int i = 0; i < M * K; ++i) h_A_fp8[i] = __nv_fp8_e4m3(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp8[i] = __nv_fp8_e4m3(h_B[i]);

    cudaMemcpy(d_A, h_A_fp8, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp8, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < ( 2 ); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = ( 10 );
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
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
    __nv_fp8_e4m3 *h_C_fp8 = new __nv_fp8_e4m3[M * N];
    cudaMemcpy(h_C_fp8, d_C, M*N*sizeof(fp8e4m3), cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = float(h_C_fp8[i]);
    }

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 0.10) { // large because of fp8 vs fp32 numerics
            if(error_count < 25) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 25) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        // if (error > max_error) printf("Error at row %d col %d: %f != %f (ref)\n", i / N, i % N, h_C[i], h_C_ref[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_fp8;
    delete[] h_B_fp8;
    delete[] h_C_fp8;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int N;

    N = 3072;
    run_benchmark<matmul_template<2,2,8>>(N, N, N);

    // N = 4096;
    // run_benchmark<matmul_template<2,2,8>>(N, N, N);

    // N = 6144;
    // run_benchmark<matmul_template<2,2,8>>(N, N, N);

    // N = 8192;
    // run_benchmark<matmul_template<2,2,4>>(N, N, N);

    // N = 12288;
    // run_benchmark<matmul_template<2,2,8>>(N, N, N);

    // N = 16384;
    // run_benchmark<matmul_template<2,2,8>>(N, N, N);

    return 0;
}

