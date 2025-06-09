#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include "prototype.dp.hpp"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state { sycl::int2 coord; };
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template <bool PERISISTENT_GRID = true>
    static inline dpct::dim3 grid(int M, int N, int K) {
        return dpct::dim3(
            PERISISTENT_GRID
                ? 132
                : M * N /
                      (M_BLOCK * N_BLOCK * layout::base_tile::num_elements));
    }
      // ThunderKittens template functions
    static inline void common_setup(common_setup_args<layout> args) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int Rblocks = args.globals.C.rows() / (M_BLOCK * 64),
            Cblocks = args.globals.C.cols() / (N_BLOCK * 64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter * item_ct1.get_group_range(2) +
                      item_ct1.get_group(2);
        if (task_id < super_rows * Cblocks) // 32*16 = 512
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M }; 
        else if (task_id < Rblocks*Cblocks) { // 512
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;  // 64
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            for (int n = 0; n < N_BLOCK; n++) 
                zero(args.state.accum[n]);
        }
        static void compute(consumer_compute_args<layout> args) {
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
        static void finish(consumer_finish_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                   {args.common.coord.x, args.common.coord.y+i});
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
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k];
                // sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

template <typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K,
               dpct::dim3 grid, dpct::dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    // printf("M: %d, N: %d, K: %d\n", M, N, K);
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    /*
    DPCT1049:367: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(MAX_SHARED_MEMORY - 1024), cgh);

            cgh.depends_on(
                dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1) {
                    prototype::lcf::kernel<mmt>(
                        G, dpct_local_acc_ct1
                               .get_multi_ptr<sycl::access::decorated::no>()
                               .get());
                });
        });
    }
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    dpct::err0 cudaStatus;

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
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    sycl::ext::oneapi::bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(sycl::ext::oneapi::bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(sycl::ext::oneapi::bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(sycl::ext::oneapi::bfloat16));

    // Check for CUDA errors
    /*
    DPCT1010:545: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaStatus = 0;
    /*
    DPCT1000:542: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (cudaStatus != 0) {
        /*
        DPCT1009:546: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:541: The statement could not be removed.
        */
        std::cerr << "CUDA error: " << dpct::get_error_string_dummy(cudaStatus)
                  << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    sycl::ext::oneapi::bfloat16 *h_A_bf16 =
        new sycl::ext::oneapi::bfloat16[M * K];
    sycl::ext::oneapi::bfloat16 *h_B_bf16 =
        new sycl::ext::oneapi::bfloat16[K * N];
    for (int i = 0; i < M * K; ++i)
        h_A_bf16[i] = sycl::ext::intel::math::float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i)
        h_B_bf16[i] = sycl::ext::intel::math::float2bfloat16(h_B[i]);

    dpct::get_in_order_queue().memcpy(d_A, h_A_bf16, M * K * 2).wait();
    dpct::get_in_order_queue().memcpy(d_B, h_B_bf16, K * N * 2).wait();

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dpct::dim3 grid(mmt::grid(M, N, K));
    dpct::dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    dpct::get_current_device().queues_wait_and_throw();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }
    dpct::get_current_device().queues_wait_and_throw();

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
    /*
    DPCT1010:547: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaStatus = 0;
    /*
    DPCT1000:544: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (cudaStatus != 0) {
        /*
        DPCT1009:548: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:543: The statement could not be removed.
        */
        std::cerr << "CUDA error: " << dpct::get_error_string_dummy(cudaStatus)
                  << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    sycl::ext::oneapi::bfloat16 *h_C_bf16 =
        new sycl::ext::oneapi::bfloat16[M * N];
    dpct::get_in_order_queue().memcpy(h_C_bf16, d_C, M * N * 2).wait();

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i)
        h_C[i] = sycl::ext::intel::math::bfloat162float(h_C_bf16[i]);

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
    dpct::dpct_free(d_A, dpct::get_in_order_queue());
    dpct::dpct_free(d_B, dpct::get_in_order_queue());
    dpct::dpct_free(d_C, dpct::get_in_order_queue());

    return 0;
}

int main() {
    int N;
    N = 4096;
    run_benchmark<matmul_template<2,4,8>>(N, N, N);
    return 0;
}