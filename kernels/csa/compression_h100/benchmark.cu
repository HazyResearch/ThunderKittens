// Timing sweep for the CSA block compression kernel over (NUM_WORKERS, C_CHUNK,
// INPUT_PIPE_STAGES). Every combo below fits H100's per-SM shared-memory budget (~230912 bytes)
// and respects NUM_WORKERS % 4 == 0, C_CHUNK <= 256 (TMA's unswizzled-tile-width cap).
#include "compression_h100.cuh"

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <limits>

struct bench_data {
    std::vector<fp8e4m3> value_a, value_b, score_a, score_b;
    std::vector<bf16> bias_a, bias_b;
};
struct bench_best {
    double us   = std::numeric_limits<double>::max();
    double gbps = 0;
    std::string desc;
};

static std::vector<fp8e4m3> random_fp8e4m3(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<fp8e4m3> out(n);
    for (size_t i = 0; i < n; i++) out[i] = fp8e4m3(dist(rng));
    return out;
}
static std::vector<bf16> random_bf16(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<bf16> out(n);
    for (size_t i = 0; i < n; i++) out[i] = __float2bfloat16(dist(rng));
    return out;
}

template<int C, int C_CHUNK, int M, int NUM_WORKERS, int PIPE>
void benchmark_one(int B, int N, const bench_data &data, bench_best &best) {
    using ker    = compression_template<C, C_CHUNK, M, NUM_WORKERS, PIPE>;
    using layout = typename ker::layout;
    int num_blocks = N / M;

    size_t value_elems = (size_t)B * N * C;
    size_t bias_elems  = (size_t)M * C;
    size_t out_elems   = (size_t)B * num_blocks * C;

    fp8e4m3 *d_value_a, *d_value_b, *d_score_a, *d_score_b;
    bf16 *d_bias_a, *d_bias_b, *d_compressed;
    cudaMalloc(&d_value_a, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_value_b, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_score_a, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_score_b, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_bias_a,  bias_elems  * sizeof(bf16));
    cudaMalloc(&d_bias_b,  bias_elems  * sizeof(bf16));
    cudaMalloc(&d_compressed, out_elems * sizeof(bf16));
    cudaMemcpy(d_value_a, data.value_a.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_b, data.value_b.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_a, data.score_a.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_b, data.score_b.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_a,  data.bias_a.data(),  bias_elems  * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_b,  data.bias_b.data(),  bias_elems  * sizeof(bf16), cudaMemcpyHostToDevice);

    typename layout::block_global Va(d_value_a, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Vb(d_value_b, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Sa(d_score_a, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Sb(d_score_b, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::bias_global  Ba(d_bias_a,  nullptr,   nullptr, nullptr,   nullptr);
    typename layout::bias_global  Bb(d_bias_b,  nullptr,   nullptr, nullptr,   nullptr);
    typename layout::out_global   Og(d_compressed, (size_t)B, nullptr, (size_t)num_blocks, nullptr);
    typename layout::globals globals = {Va, Vb, Sa, Sb, Ba, Bb, Og, /*chunk_idx=*/0};

    unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 2000;
    cudaFuncSetAttribute(prototype::lcf::kernel<ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaGetLastError(); // clear, checked properly below

    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker>;
    dim3 grid(132, 1, 1);

    constexpr int WARMUP = 5;
    constexpr int ITERS  = 20;

    auto run_one_pass = [&]() {
        for (int chunk_idx = 0; chunk_idx < layout::NUM_CHUNKS; chunk_idx++) {
            globals.chunk_idx = chunk_idx;
            prototype::lcf::kernel<ker><<<grid, BLOCK_SIZE, mem_size>>>(globals);
        }
    };

    for (int i = 0; i < WARMUP; i++) run_one_pass();
    cudaDeviceSynchronize();

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) run_one_pass();
    cudaDeviceSynchronize();
    const auto finish = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaGetLastError();
    double avg_us = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / (double)ITERS;

    // Total bytes moved per full pass (all NUM_CHUNKS launches together): the 4 input
    // tensors (fp8e4m3, 1 byte/elem) read once each across their full width, plus the
    // output (bf16, 2 bytes/elem) written once.
    double bytes_per_pass = 4.0 * value_elems * sizeof(fp8e4m3) + (double)out_elems * sizeof(bf16);
    double gbps = (bytes_per_pass / 1e9) / (avg_us / 1e6);

    std::cout << "NUM_WORKERS=" << NUM_WORKERS << " C_CHUNK=" << C_CHUNK << " PIPE=" << PIPE
               << " (NUM_CHUNKS=" << layout::NUM_CHUNKS << ") -> "
               << avg_us << " us/pass, " << gbps << " GB/s";
    if (err != cudaSuccess) {
        std::cout << "  [ERROR: " << cudaGetErrorString(err) << "]";
    } else if (avg_us < best.us) {
        best.us   = avg_us;
        best.gbps = gbps;
        best.desc = "NUM_WORKERS=" + std::to_string(NUM_WORKERS) +
                    " C_CHUNK=" + std::to_string(C_CHUNK) +
                    " PIPE=" + std::to_string(PIPE);
    }
    std::cout << "\n";

    cudaFree(d_value_a); cudaFree(d_value_b); cudaFree(d_score_a); cudaFree(d_score_b);
    cudaFree(d_bias_a);  cudaFree(d_bias_b);  cudaFree(d_compressed);
}

int main() {
    constexpr int C_FULL = 512, M = 4;
    constexpr int B = 8, N = 32768;

    std::cout << "Sweeping NUM_WORKERS x C_CHUNK x INPUT_PIPE_STAGES, C=" << C_FULL
              << " M=" << M << " B=" << B << " N=" << N << "\n\n";

    std::mt19937 rng(42);
    size_t value_elems = (size_t)B * N * C_FULL;
    size_t bias_elems  = (size_t)M * C_FULL;
    bench_data data{
        random_fp8e4m3(value_elems, rng), random_fp8e4m3(value_elems, rng),
        random_fp8e4m3(value_elems, rng), random_fp8e4m3(value_elems, rng),
        random_bf16(bias_elems, rng), random_bf16(bias_elems, rng),
    };
    bench_best best;

    benchmark_one<C_FULL, 32, M, 4, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 4, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 4, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 4, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 4, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 4, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 4, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 4, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 4, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 4, 2>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 4, 3>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 4, 4>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 4, 1>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 4, 2>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 4, 3>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 4, 4>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 8, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 8, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 8, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 8, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 8, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 8, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 8, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 8, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 8, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 8, 2>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 8, 3>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 8, 4>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 8, 1>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 8, 2>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 8, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 12, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 12, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 12, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 12, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 12, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 12, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 12, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 12, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 12, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 12, 2>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 12, 3>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 12, 4>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 12, 1>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 12, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 16, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 16, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 16, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 16, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 16, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 16, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 16, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 16, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 16, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 16, 2>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 16, 3>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 16, 1>(B, N, data, best);

    benchmark_one<C_FULL, 32, M, 20, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 20, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 20, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 20, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 20, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 20, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 20, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 20, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 20, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 20, 2>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 20, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 24, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 24, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 24, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 24, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 24, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 24, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 24, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 24, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 24, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 24, 2>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 24, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 28, 1>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 28, 2>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 28, 3>(B, N, data, best);
    benchmark_one<C_FULL, 32, M, 28, 4>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 28, 1>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 28, 2>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 28, 3>(B, N, data, best);
    benchmark_one<C_FULL, 64, M, 28, 4>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 28, 1>(B, N, data, best);
    benchmark_one<C_FULL, 128, M, 28, 2>(B, N, data, best);
    benchmark_one<C_FULL, 256, M, 28, 1>(B, N, data, best);

    std::cout << "\nBEST: " << best.desc << " -> " << best.us << " us/pass, " << best.gbps << " GB/s\n";
    return 0;
}
