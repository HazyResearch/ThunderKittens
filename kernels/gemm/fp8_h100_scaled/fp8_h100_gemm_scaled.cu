#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <omp.h>

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using c_dtype = float;

struct matmul_layout {
    // tiles for the quantized inputs
    using  a_tile   = st_fp8e4m3<64, 128>; 
    using  b_tile   = st_fp8e4m3<128, 128>;
    using  c_tile   = st<c_dtype, 64, 128>;
    using  a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout = gl<c_dtype, 1, 1, -1, -1, c_tile>;

    // tiles for the dequantized inputs
    using scale_a_layout = gl<c_dtype, 1, 1, 1, -1>;
    using scale_b_layout = gl<c_dtype, 1, 1, 1, -1>;

    template<typename T=float> using accum_tile = rt<T, 16, c_tile::cols>;

    struct globals        { 
        a_layout A; b_layout B; c_layout C; 
        scale_a_layout scale_a; scale_b_layout scale_b;
    };

    struct input_block    { 
        a_tile a[2]; b_tile b; 
    };
    struct finish_block   { 
        c_tile c[2]; 
    };
    struct scratch_block  {
    };
    struct common_state   { int2 coord; };
    struct consumer_state { 
        accum_tile<c_dtype> accum;      // Changed to single tall accumulator
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout    = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(2*layout::c_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
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
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                #pragma unroll
                for(int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B,
                                {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            warp::zero(args.state.accum); 
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            col_vec<rt<c_dtype, 16, 128>> scale_a_rv;
            row_vec<rt<c_dtype, 16, 128>> scale_b_rv;
            warpgroup::load(scale_a_rv, args.globals.scale_a, {args.common.coord.x});
            warp::load(scale_b_rv, args.globals.scale_b, {args.common.coord.y});
            warp::mul_col(args.state.accum, args.state.accum, scale_b_rv);
            warp::mul_row(args.state.accum, args.state.accum, scale_a_rv);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                 {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(
    fp8e4m3 *d_A, fp8e4m3 *d_B, c_dtype *d_C, 
    c_dtype *d_scale_a, c_dtype *d_scale_b,
    size_t M, size_t N, size_t K, 
    dim3 grid, dim3 block
) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};

    // scales
    using scale_a_layout = typename mmt::layout::scale_a_layout;
    using scale_b_layout = typename mmt::layout::scale_b_layout;
    scale_a_layout scale_a{d_scale_a, nullptr, nullptr, nullptr, M};
    scale_b_layout scale_b{d_scale_b, nullptr, nullptr, nullptr, N};

    globals G{Ag, Bg, Cg, scale_a, scale_b};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

void write_matrix_to_csv(const std::string& filename, float* matrix, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << matrix[i * cols + j];
            if (j < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    std::cout << "CPU M=" << M << " N=" << N << " K=" << K << std::endl;
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k]; // mma_ABt
            }
            c[i * N + j] = sum;
        }
    }
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution dis(0.0f, 1.0f);

    // Initialize matrices with random values
    // for (int i = 0; i < M * K; ++i) h_A[i] = i / 100000.0f;  // dis(gen) * 0.2f; 
    // for (int i = 0; i < K * N; ++i) h_B[i] = i / 100000.0f;   // dis(gen) * 0.2f; 
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen) * 0.2f; 
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen) * 0.2f; 

    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    fp8e4m3 *d_A, *d_B;
    c_dtype *d_C;
    cudaMalloc(&d_A, M*K*sizeof(fp8e4m3));
    cudaMalloc(&d_B, K*N*sizeof(fp8e4m3));
    cudaMalloc(&d_C, M*N*sizeof(c_dtype));
    // scales
    c_dtype *d_scale_a, *d_scale_b;
    cudaMalloc(&d_scale_a, M*sizeof(c_dtype));
    cudaMalloc(&d_scale_b, N*sizeof(c_dtype));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
    std::cout << "Performed CPU matrix multiplication" << std::endl;

    //  Obtain inputs on GPU device
    const float FP8_E4M3_MAX = 448.0f;
    // const float FP8_E4M3_MIN = -448.0f;
    c_dtype *h_scale_a = new c_dtype[M];
    c_dtype *h_scale_b = new c_dtype[N];
    __nv_fp8_e4m3 *h_A_fp8_scaled = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8_scaled = new __nv_fp8_e4m3[K * N];
    
    // row-wise scaling
    for(int row = 0; row < M; row++) {
        float max_val = 0.0f;
        for(int col = 0; col < K; col++) {
            float abs_val = std::abs(h_A[row * K + col]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_a[row] = c_dtype(max_val / FP8_E4M3_MAX); 
        if ( row < 10 ) {
            std::cout << "h_scale_a[" << row << "] = " << float(h_scale_a[row]) << ", max_val: " << max_val << std::endl;
        }
    }

    // fill h_A_fp8_scaled by following to_float8_e4m3fn. 
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            h_A_fp8_scaled[i * K + j] = __nv_fp8_e4m3(h_A[i * K + j] / float(h_scale_a[i]));
        }
    }

    // column-wise scaling
    for(int col = 0; col < N; col++) {
        float max_val = 0.0f;
        for(int row = 0; row < K; row++) {
            float abs_val = std::abs(h_B[row + col*K]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_b[col] = c_dtype(max_val / FP8_E4M3_MAX);

        if ( col < 10 ) {
            std::cout << "h_scale_b[" << col << "] = " << float(h_scale_b[col]) << ", max_val: " << max_val << std::endl;
        }
    }

    // fill h_B_fp8_scaled by following to_float8_e4m3fn
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            h_B_fp8_scaled[j + i * K] = __nv_fp8_e4m3(h_B[j + i * K] / float(h_scale_b[i]));
        }
    }
    
    cudaMemcpy(d_A, h_A_fp8_scaled, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp8_scaled, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_a, h_scale_a, M*sizeof(c_dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_b, h_scale_b, N*sizeof(c_dtype), cudaMemcpyHostToDevice);

    /* 
    Launch kernel
    */
    std::cout << "Copied matrices to device" << std::endl;
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < ( 2 ); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block); 
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = ( 10 );
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block); 
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
    c_dtype *h_C_out = new c_dtype[M * N];
    cudaMemcpy(h_C_out, d_C, M*N*sizeof(c_dtype), cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = float(h_C_out[i]);
    }

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f, total_error = 0.0f, total_ref = 0.0f, total_ours=0.0f;
    float input_a = 0.0f, input_b = 0.0f;
    int error_count = 0;
    printf("Num rows: %zu, Num cols: %zu\n", M, N);
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if( error > 0.7f ) { // large because of fp8 vs fp32 numerics # error > 0.10
            if(error_count < 10) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 700) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
        total_ref += std::abs(h_C_ref[i]);
        total_error += error;
        total_ours += std::abs(h_C[i]);
    }

    for (int i = 0; i < M * K; i++) {
        input_a += std::abs(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        input_b += std::abs(h_B[i]);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Average error: " << total_error / M / N << std::endl;
    std::cout << "Average ref: " << total_ref / (M * N) << std::endl;
    std::cout << "Average ours: " << total_ours / M / N << std::endl;
    std::cout << "Average input_a: " << input_a / M / K << std::endl;
    std::cout << "Average input_b: " << input_b / K / N << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
 
    // write_matrix_to_csv("h_C_ref.csv", h_C_ref, M, N);
    // write_matrix_to_csv("h_C.csv", h_C, M, N);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_C_out;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    run_benchmark<matmul_template<8>>(M, N, K);
    return 0;
}
