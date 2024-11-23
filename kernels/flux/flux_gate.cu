#define RUN_MAIN
// #define TORCH_COMPILE_GATE

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
template<typename op, kittens::ducks::sv::all SV> __device__ static inline void rt_sv_op(rt_fl<16,SV::length> &acc, const SV &bias) {
    #pragma unroll
    for(int i = 0; i < SV::tiles; i++) {
        float2 tmp1 = __half22float2(*(half_2*)&bias.data[16*i + 0 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[0] = op::template op<float2>(acc.tiles[0][i].data[0], tmp1);
        acc.tiles[0][i].data[1] = op::template op<float2>(acc.tiles[0][i].data[1], tmp1);
        float2 tmp2 = __half22float2(*(half_2*)&bias.data[16*i + 8 + 2*(laneid()%4)]);
        acc.tiles[0][i].data[2] = op::template op<float2>(acc.tiles[0][i].data[2], tmp2);
        acc.tiles[0][i].data[3] = op::template op<float2>(acc.tiles[0][i].data[3], tmp2);
    }
}
template<typename op, kittens::ducks::st::all ST> __device__ static inline void wg_rt_sv_op(rt_fl<16,ST::cols> &acc, const ST &y) {
    static_assert(ST::rows == 64);
    #pragma unroll
    for(int i = 0; i < ST::cols; i++) {
        acc.tiles[0][i].data[0] = op::template op<float2>(
            acc.tiles[0][i].data[0],
            __half22float2(*(half_2*)&y[{warpgroup::warpid()*16 + 0 + laneid()/4, 16*i + 0 + 2*(laneid()%4)}]));
        acc.tiles[0][i].data[1] = op::template op<float2>(
            acc.tiles[0][i].data[1],
            __half22float2(*(half_2*)&y[{warpgroup::warpid()*16 + 8 + laneid()/4, 16*i + 0 + 2*(laneid()%4)}]));
        acc.tiles[0][i].data[2] = op::template op<float2>(
            acc.tiles[0][i].data[2],
            __half22float2(*(half_2*)&y[{warpgroup::warpid()*16 + 0 + laneid()/4, 16*i + 8 + 2*(laneid()%4)}]));
        acc.tiles[0][i].data[3] = op::template op<float2>(
            acc.tiles[0][i].data[3],
            __half22float2(*(half_2*)&y[{warpgroup::warpid()*16 + 8 + laneid()/4, 16*i + 8 + 2*(laneid()%4)}]));
    }
}
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int _transpose_lhs, int _transpose_rhs>
struct flux_matmul_gate_layout {
    constexpr static bool transpose_lhs = _transpose_lhs, transpose_rhs = _transpose_rhs;
    using lhs_tile  = std::conditional_t<transpose_lhs, st_hf<BLOCK_K,      64>, st_hf<     64, BLOCK_K>>;
    using rhs_tile  = std::conditional_t<transpose_rhs, st_hf<BLOCK_N, BLOCK_K>, st_hf<BLOCK_K, BLOCK_N>>;
    using acc_tile  = st_hf<64, BLOCK_N>;
    using bias_vec  = sv_hf<acc_tile::cols>;
    struct globals { // global layout (here with TMA descriptors)
        gl<half, 1, 1, -1, -1, lhs_tile> lhs;
        gl<half, 1, 1, -1, -1, rhs_tile> rhs;
        gl<half, 1, 1,  1, -1, bias_vec> bias;
        gl<half, 1, 1,  1, -1, bias_vec> gate;
        gl<half, 1, 1, -1, -1, acc_tile> y;
        gl<half, 1, 1, -1, -1, acc_tile> acc;
    };
    struct input_block {
        lhs_tile lhs[BLOCK_M/64];
        rhs_tile rhs;
    };
    struct scratch_block  { bias_vec bias, gate; };
    struct consumer_state { rt_fl<16, BLOCK_N> acc;   };
    struct finish_block   { acc_tile acc[BLOCK_M/64]; };
};
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int _transpose_lhs=0, int _transpose_rhs=0>
struct flux_matmul_gate_template {
    constexpr static bool transpose_lhs = _transpose_lhs, transpose_rhs = _transpose_rhs;
    using layout = flux_matmul_gate_layout<BLOCK_M, BLOCK_N, BLOCK_K, transpose_lhs, transpose_rhs>;
    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4,
                         INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == 0) {
            args.num_iters = transpose_lhs ? args.globals.lhs.rows / BLOCK_K : args.globals.lhs.cols / BLOCK_K;
        }
        else args.num_iters = -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
            warpgroup::producer_registers(); // decrease registers for the producer warpgroup
        }
        __device__ static void load(producer_load_args<layout> args) { // semaphore for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(args.inputs_arrived, sizeof(layout::input_block));
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    if constexpr (transpose_lhs)
                        tma::load_async(args.input.lhs[i], args.globals.lhs, {args.iter, (int)blockIdx.x*NUM_CONSUMER_WARPGROUPS+i}, args.inputs_arrived);
                    else
                        tma::load_async(args.input.lhs[i], args.globals.lhs, {(int)blockIdx.x*NUM_CONSUMER_WARPGROUPS+i, args.iter}, args.inputs_arrived);
                }
                if constexpr (transpose_rhs)
                    tma::load_async(args.input.rhs, args.globals.rhs, {(int)blockIdx.y, args.iter}, args.inputs_arrived);
                else
                    tma::load_async(args.input.rhs, args.globals.rhs, {args.iter, (int)blockIdx.y}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
            warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {blockIdx.y});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.gate, args.globals.gate, {blockIdx.y});
            group<NUM_CONSUMER_WARPS>::sync(6);
            rt_sv_op<base_ops::copy2>(args.state.acc, args.scratch.bias); // copy bias in to start
            zero(args.state.acc);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            if constexpr (transpose_lhs && transpose_rhs)
                warpgroup::mma_AtBt(args.state.acc, args.input.lhs[warpgroup::groupid()], args.input.rhs);
            else if constexpr (transpose_lhs)
                warpgroup::mma_AtB (args.state.acc, args.input.lhs[warpgroup::groupid()], args.input.rhs);
            else if constexpr (transpose_rhs)
                warpgroup::mma_ABt (args.state.acc, args.input.lhs[warpgroup::groupid()], args.input.rhs);
            else
                warpgroup::mma_AB  (args.state.acc, args.input.lhs[warpgroup::groupid()], args.input.rhs);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            kittens::coord idx = {blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), blockIdx.y};
            warpgroup::load(args.finish.acc[warpgroup::groupid()], args.globals.y, idx);
            rt_sv_op<base_ops::mul>(args.state.acc, args.scratch.gate); // multiply gate onto acc
            warpgroup::sync(6); // y now arrived
            wg_rt_sv_op<base_ops::sum>(args.state.acc, args.finish.acc[warpgroup::groupid()]); // add y onto acc
            warpgroup::store(args.finish.acc[warpgroup::groupid()], args.state.acc); // now that we're done with that, store the result into same slot.
            warpgroup::sync(6);
            if(warpgroup::warpid() == 0) {
                tma::store_async(args.globals.acc, args.finish.acc[warpgroup::groupid()], idx);
                tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.finish_finished);
            }
        }
    };
};

#include <iostream>
#include <random>

#ifdef RUN_MAIN
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <omp.h>
template<int transpose_lhs, int transpose_rhs>
void cpu_gemm(float* a, float* b, float *bias, float *gate, float *y, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                if constexpr (transpose_lhs && transpose_rhs) {
                    sum += a[k * M + i] * b[j * K + k];
                } else if constexpr (transpose_lhs) {
                    sum += a[k * M + i] * b[k * N + j];
                } else if constexpr (transpose_rhs) {
                    sum += a[i * K + k] * b[j * K + k];
                } else {
                    sum += a[i * K + k] * b[k * N + j];
                }
            }
            c[i * N + j] = (sum + bias[j]) * gate[j] + y[i*N + j];
        }
    }
}

template<typename fmt>
int run_bench(int M, int N, int K) {

    using lhs_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::input_block>().lhs[0])>::type;
    using rhs_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::input_block>().rhs)>::type;
    using acc_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::finish_block>().acc[0])>::type;
    using lhs_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().lhs)>::type;
    using rhs_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().rhs)>::type;
    using bias_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().bias)>::type;
    using acc_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().acc)>::type;
    using globals  = typename fmt::layout::globals;

    static constexpr int transpose_lhs = fmt::transpose_lhs, transpose_rhs = fmt::transpose_rhs;
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_bias = new float[N];
    float *h_gate = new float[N];
    float *h_y = new float[M * N];
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
    for (int i = 0; i < N; ++i) h_bias[i] = dis(gen);
    for (int i = 0; i < N; ++i) h_gate[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) h_y[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    // cpu_gemm<transpose_lhs, transpose_rhs>(h_A, h_B, h_bias, h_gate, h_y, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_bias, *d_gate, *d_y;
    cudaMalloc(&d_A, M*K*2);
    cudaMalloc(&d_B, K*N*2);
    cudaMalloc(&d_C, M*N*2);
    cudaMalloc(&d_bias, N*2);
    cudaMalloc(&d_gate, N*2);
    cudaMalloc(&d_y, M*N*2);
    std::cout << "Allocated device memory" << std::endl;

    std::cout << "lhs_tile::rows=" << lhs_tile::rows << " lhs_tile::cols=" << lhs_tile::cols << std::endl;
    std::cout << "rhs_tile::rows=" << rhs_tile::rows << " rhs_tile::cols=" << rhs_tile::cols << std::endl;
    std::cout << "acc_tile::rows=" << acc_tile::rows << " acc_tile::cols=" << acc_tile::cols << std::endl;
    lhs_global Ag{d_A, nullptr, nullptr, transpose_lhs ? K : M, transpose_lhs ? M : K};
    rhs_global Bg{d_B, nullptr, nullptr, transpose_rhs ? N : K, transpose_rhs ? K : N};
    acc_global Cg{d_C, nullptr, nullptr, M, N};
    acc_global Yg{d_y, nullptr, nullptr, M, N};
    bias_global Biasg{d_bias, nullptr, nullptr, nullptr, N};
    bias_global Gateg{d_gate, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Biasg, Gateg, Yg, Cg};

    std::cout << "Allocated memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_half = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_half = new __nv_bfloat16[K * N];
    __nv_bfloat16 *h_bias_half = new __nv_bfloat16[N];
    __nv_bfloat16 *h_gate_half = new __nv_bfloat16[N];
    __nv_bfloat16 *h_y_half = new __nv_bfloat16[M * N];
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2bfloat16(h_B[i]);
    for (int i = 0; i < N; ++i) h_bias_half[i] = __float2bfloat16(h_bias[i]);
    for (int i = 0; i < N; ++i) h_gate_half[i] = __float2bfloat16(h_gate[i]);
    for (int i = 0; i < M * N; ++i) h_y_half[i] = __float2bfloat16(h_y[i]);

    cudaMemcpy(d_A, h_A_half, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, K*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias_half, N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate_half, N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y_half, M*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(prototype::lcf::kernel<fmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // Launch kernel
    dim3 grid(M / (acc_tile::rows*prototype::detail::NUM_CONSUMER_WARPGROUPS_v<fmt>), N / acc_tile::cols); // rows, cols
    dim3 block(prototype::detail::NUM_THREADS_v<fmt>);

    // Start timing
    std::cout << "Warming up kernel" << std::endl;
    prototype::lcf::kernel<fmt><<<grid, block, mem_size>>>(G); // warmup
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/lhs_tile::cols << " reduction block dimension\n";
    std::cout << "Kernel has " << kittens::prototype::detail::INPUT_PIPE_STAGES_v<fmt> << " input pipeline stages and " << kittens::prototype::detail::OUTPUT_PIPE_STAGES_v<fmt> << " output pipeline stages\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 10;
    for(int i = 0; i < ITERS; i++) {
        prototype::lcf::kernel<fmt><<<grid, block, mem_size>>>(G);
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

    std::cout << "Kernel execution time: " << seconds*1000 << " ms\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_half = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_half, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    // for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_half[i]);

    // std::cout << "Converted result back to float" << std::endl;

    // // Check result
    // float max_error = 0.0f;
    // int error_count = 0;
    // for (int i = 0; i < M * N; ++i) {
    //     float error = std::abs(h_C[i] - h_C_ref[i]);
    //     if(error > 1.0) { // large because of half vs fp32 numerics
    //         if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
    //         else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
    //         error_count++;
    //     }
    //     max_error = std::max(max_error, error);
    // }

    // std::cout << "Max error: " << max_error << std::endl;
    // std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_half;
    delete[] h_B_half;
    delete[] h_C_half;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    constexpr int transpose_lhs = 0, transpose_rhs = 1;
    run_bench<flux_matmul_gate_template<192, 192, 64>>(192*12, 192*11, 8192);
    run_bench<flux_matmul_gate_template<128, 256, 64>>(128*12, 256*11, 8192);
    // run_bench<flux_matmul_gate_template<128, 14*16, 64>>(4096, 4096*7/8, 4096);
    // run_bench<flux_matmul_gate_template<128, 256, 64>>(2048, 2048, 2048*7);
}
#endif

#ifdef TORCH_COMPILE_GATE
#include "pyutils/torch_helpers.cuh"
#include <iostream>

template<int M_tile, int K_tile, int N_tile, int transpose_lhs, int transpose_rhs>
void dispatch_fused_flux_linear_gate(
    half * d_x,
    half * d_weight,
    half * d_bias,
    half * d_gate,
    half * d_y,
    half * d_out,
    uint M, uint K, uint N
) {
    using fmt = flux_matmul_gate_template<M_tile, K_tile, N_tile, transpose_lhs, transpose_rhs>;

    using lhs_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::input_block>().lhs[0])>::type;
    using rhs_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::input_block>().rhs)>::type;
    using acc_tile   = typename std::remove_reference<decltype(std::declval<typename fmt::layout::finish_block>().acc[0])>::type;
    using lhs_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().lhs)>::type;
    using rhs_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().rhs)>::type;
    using bias_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().bias)>::type;
    using acc_global = typename std::remove_reference<decltype(std::declval<typename fmt::layout::globals>().acc)>::type;
    using globals  = typename fmt::layout::globals;

    lhs_global Ag{d_x, nullptr, nullptr, transpose_lhs ? K : M, transpose_lhs ? M : K};
    rhs_global Bg{d_weight, nullptr, nullptr, transpose_rhs ? N : K, transpose_rhs ? K : N};
    acc_global Cg{d_out, nullptr, nullptr, M, N};
    acc_global Yg{d_y, nullptr, nullptr, M, N};
    bias_global Biasg{d_bias, nullptr, nullptr, nullptr, N};
    bias_global Gateg{d_gate, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Biasg, Gateg, Yg, Cg};

    unsigned long mem_size = MAX_SHARED_MEMORY; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(prototype::lcf::kernel<fmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // Launch kernel
    dim3 grid(M / (acc_tile::rows*prototype::detail::NUM_CONSUMER_WARPGROUPS_v<fmt>), N / acc_tile::cols); // rows, cols
    dim3 block(prototype::detail::NUM_THREADS_v<fmt>);

    prototype::lcf::kernel<fmt><<<grid, block, mem_size>>>(G);
}

torch::Tensor fused_flux_linear_gate(
    const torch::Tensor x,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor gate,
    const torch::Tensor y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(gate);
    CHECK_INPUT(y);

    const uint M = x.size(0);
    const uint K = x.size(1);
    const uint N = weight.size(0);
    
    TORCH_CHECK(weight.size(1) == K, "weight has incompatible shape");
    TORCH_CHECK(bias.size(0) == N, "bias has incompatible shape");
    TORCH_CHECK(gate.size(0) == N, "gate has incompatible shape");
    TORCH_CHECK(y.size(0) == M, "y has incompatible shape");
    TORCH_CHECK(y.size(1) == N, "y has incompatible shape");

    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be in N x K format");

    // // x contiguous means x is M x K format, so transpose_lhs = false
    // const bool transpose_lhs = !x.is_contiguous();
    // // weight contiguous means weight is in N x K format, so transpose_rhs = true!
    // const bool transpose_rhs = weight.is_contiguous();

    torch::Tensor out = torch::empty({M, N}, y.options());

    // convert to half
    c10::BFloat16 *x_half = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *weight_half = weight.data_ptr<c10::BFloat16>();
    c10::BFloat16 *bias_half = bias.data_ptr<c10::BFloat16>();
    c10::BFloat16 *gate_half = gate.data_ptr<c10::BFloat16>();
    c10::BFloat16 *y_half = y.data_ptr<c10::BFloat16>();
    c10::BFloat16 *out_half = out.data_ptr<c10::BFloat16>();

    half *d_x = reinterpret_cast<half*>(x_half);
    half *d_weight = reinterpret_cast<half*>(weight_half);
    half *d_bias = reinterpret_cast<half*>(bias_half);
    half *d_gate = reinterpret_cast<half*>(gate_half);
    half *d_y = reinterpret_cast<half*>(y_half);
    half *d_out = reinterpret_cast<half*>(out_half);

    if (M > 512) {
        const int M_tile = 192;
        const int K_tile = 192;
        const int N_tile = 64;

        dispatch_fused_flux_linear_gate<M_tile, K_tile, N_tile, false, false>(d_x, d_weight, d_bias, d_gate, d_y, d_out, M, K, N);
    } else if (M > 256 && K > 3072) {
        const int M_tile = 128;
        const int K_tile = 96;
        const int N_tile = 128;
        
        dispatch_fused_flux_linear_gate<M_tile, K_tile, N_tile, false, true>(d_x, d_weight, d_bias, d_gate, d_y, d_out, M, K, N);
    } else if (K > 3072) {
        const int M_tile = 64;
        const int K_tile = 96;
        const int N_tile = 128;
        
        dispatch_fused_flux_linear_gate<M_tile, K_tile, N_tile, false, true>(d_x, d_weight, d_bias, d_gate, d_y, d_out, M, K, N);
    } else {
        const int M_tile = 128;
        const int K_tile = 192;
        const int N_tile = 64;
        
        dispatch_fused_flux_linear_gate<M_tile, K_tile, N_tile, false, true>(d_x, d_weight, d_bias, d_gate, d_y, d_out, M, K, N);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());

    return out;
}

#endif