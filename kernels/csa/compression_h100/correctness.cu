// Correctness test for the CSA block compression kernel
#include "compression_h100.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaCheckError() failed at %s:%d : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%d : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

static void read_floats(std::ifstream &infile, std::vector<float> &out) {
    for (size_t i = 0; i < out.size(); i++) infile >> out[i];
}
static std::vector<bf16> to_bf16(const std::vector<float> &in) {
    std::vector<bf16> out(in.size());
    for (size_t i = 0; i < in.size(); i++) out[i] = __float2bfloat16(in[i]);
    return out;
}
// gentests.py already rounds these to fp8e4m3-representable values, so this cast is exact.
static std::vector<fp8e4m3> to_fp8e4m3(const std::vector<float> &in) {
    std::vector<fp8e4m3> out(in.size());
    for (size_t i = 0; i < in.size(); i++) out[i] = fp8e4m3(in[i]);
    return out;
}

// Runs one (C, C_CHUNK, M) instantiation against a fixture read from `infile`. Read order
// must match gentests.py's write order: value_a, value_b, score_a, score_b, bias_a,
// bias_b, ref_compressed. Returns true if the kernel output is within tolerance.
template<int C, int C_CHUNK, int M, int NUM_WORKERS=4, int PIPE=2>
bool run_case(std::ifstream &infile, int B, int N) {
    using ker    = compression_template<C, C_CHUNK, M, NUM_WORKERS, PIPE>;
    using layout = typename ker::layout;
    int num_blocks = N / M;

    size_t value_elems = (size_t)B * N * C;
    size_t bias_elems  = (size_t)M * C;
    size_t out_elems   = (size_t)B * num_blocks * C;

    std::vector<float> h_value_a(value_elems), h_value_b(value_elems);
    std::vector<float> h_score_a(value_elems), h_score_b(value_elems);
    std::vector<float> h_bias_a(bias_elems),   h_bias_b(bias_elems);
    std::vector<float> h_ref(out_elems);

    read_floats(infile, h_value_a);
    read_floats(infile, h_value_b);
    read_floats(infile, h_score_a);
    read_floats(infile, h_score_b);
    read_floats(infile, h_bias_a);
    read_floats(infile, h_bias_b);
    read_floats(infile, h_ref);

    auto value_a_fp8 = to_fp8e4m3(h_value_a), value_b_fp8 = to_fp8e4m3(h_value_b);
    auto score_a_fp8 = to_fp8e4m3(h_score_a), score_b_fp8 = to_fp8e4m3(h_score_b);
    auto bias_a_bf   = to_bf16(h_bias_a),     bias_b_bf   = to_bf16(h_bias_b);

    fp8e4m3 *d_value_a, *d_value_b, *d_score_a, *d_score_b;
    bf16 *d_bias_a, *d_bias_b, *d_compressed;
    cudaMalloc(&d_value_a, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_value_b, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_score_a, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_score_b, value_elems * sizeof(fp8e4m3));
    cudaMalloc(&d_bias_a,  bias_elems  * sizeof(bf16));
    cudaMalloc(&d_bias_b,  bias_elems  * sizeof(bf16));
    cudaMalloc(&d_compressed, out_elems * sizeof(bf16));

    cudaMemcpy(d_value_a, value_a_fp8.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_b, value_b_fp8.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_a, score_a_fp8.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_b, score_b_fp8.data(), value_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_a,  bias_a_bf.data(),   bias_elems  * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_b,  bias_b_bf.data(),   bias_elems  * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compile-time-fixed gl dims take nullptr, runtime ones take a real size_t.
    typename layout::block_global Va(d_value_a, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Vb(d_value_b, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Sa(d_score_a, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::block_global Sb(d_score_b, (size_t)B, nullptr, (size_t)N, nullptr);
    typename layout::bias_global  Ba(d_bias_a,  nullptr,   nullptr, nullptr,   nullptr); // all 4 dims fixed
    typename layout::bias_global  Bb(d_bias_b,  nullptr,   nullptr, nullptr,   nullptr);
    typename layout::out_global   Og(d_compressed, (size_t)B, nullptr, (size_t)num_blocks, nullptr);

    typename layout::globals globals = {Va, Vb, Sa, Sb, Ba, Bb, Og, /*chunk_idx=*/0}; // matches globals' declared field order

    unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 2000;
    cudaFuncSetAttribute(prototype::lcf::kernel<ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    CudaCheckError();

    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker>;
    dim3 grid(132, 1, 1); // extra blocks (batch >= B) exit harmlessly on their first task_iter
    // One launch per C_CHUNK-wide slice; each writes its own disjoint slice of `compressed`.
    for (int chunk_idx = 0; chunk_idx < layout::NUM_CHUNKS; chunk_idx++) {
        globals.chunk_idx = chunk_idx;
        prototype::lcf::kernel<ker><<<grid, BLOCK_SIZE, mem_size>>>(globals);
        CudaCheckError();
    }

    std::vector<bf16> out_bf(out_elems);
    cudaMemcpy(out_bf.data(), d_compressed, out_elems * sizeof(bf16), cudaMemcpyDeviceToHost);

    float total_diff = 0.f, max_diff = 0.f;
    for (size_t i = 0; i < out_elems; i++) {
        float val  = __bfloat162float(out_bf[i]);
        float diff = std::abs(val - h_ref[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
    }
    float avg_diff = total_diff / out_elems;
    bool good = avg_diff < 0.02f && max_diff < 0.15f && !std::isnan(total_diff);
    std::cout << "  [C=" << C << ", C_CHUNK=" << C_CHUNK << ", M=" << M << "] avg diff: " << avg_diff
              << ", max diff: " << max_diff << " -> " << (good ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_value_a); cudaFree(d_value_b); cudaFree(d_score_a); cudaFree(d_score_b);
    cudaFree(d_bias_a);  cudaFree(d_bias_b);  cudaFree(d_compressed);

    return good;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <fixture.txt>  (generate one with gentests.py)\n";
        return 1;
    }
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "could not open " << argv[1] << "\n";
        return 1;
    }

    // Must match gentests.py, and NUM_WORKERS/PIPE must match csa_compression_kv/indexer's
    // actual shipped values (compression_h100.cuh)
    constexpr int B = 2, N = 316;

    std::cout << "C=512, C_CHUNK=128, M=4, NUM_WORKERS=28, PIPE=1 (value/score fp8e4m3, bias/output bf16)\n";
    bool good = run_case<512, /*C_CHUNK=*/128, /*M=*/4, /*NUM_WORKERS=*/28, /*PIPE=*/1>(infile, B, N);
    std::cout << (good ? "PASSED\n" : "FAILED\n");
    return good ? 0 : 1;
}
