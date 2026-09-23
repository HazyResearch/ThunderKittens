#include "h100_lcsf_bwd.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

template <int HEAD_DIM>
void benchmark_flash_attn_bwd(int batch_size, int num_q_heads, int num_kv_heads,
                              int seq_len, int num_iters = 100) {
  size_t q_size = batch_size * num_q_heads * seq_len * HEAD_DIM;
  size_t k_size = batch_size * num_kv_heads * seq_len * HEAD_DIM;
  size_t v_size = k_size;
  size_t dq_size = q_size;
  size_t dk_size = k_size;
  size_t dv_size = v_size;
  size_t l_size = batch_size * num_q_heads * seq_len * seq_len;
  size_t d_size = l_size;

  // Allocate device memory
  __nv_bfloat16 *q_d, *k_d, *v_d, *dO_d;
  float *dq_d, *dk_d, *dv_d, *l_d, *d_d;

  cudaMalloc(&q_d, q_size * sizeof(__nv_bfloat16));
  cudaMalloc(&k_d, k_size * sizeof(__nv_bfloat16));
  cudaMalloc(&v_d, v_size * sizeof(__nv_bfloat16));
  cudaMalloc(&dO_d, q_size * sizeof(__nv_bfloat16));
  cudaMalloc(&dq_d, dq_size * sizeof(float));
  cudaMalloc(&dk_d, dk_size * sizeof(float));
  cudaMalloc(&dv_d, dv_size * sizeof(float));
  cudaMalloc(&l_d, l_size * sizeof(float));
  cudaMalloc(&d_d, d_size * sizeof(float));

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sms = prop.multiProcessorCount;

  // Warmup
  FlashAttentionBwd<HEAD_DIM>::run(q_d, dO_d, k_d, v_d, dq_d, dk_d, dv_d, l_d,
                                   d_d, batch_size, num_q_heads, num_kv_heads,
                                   seq_len, 0, num_sms);

  // Benchmark
  float total_time = 0;
  cudaEventRecord(start);

  for (int i = 0; i < num_iters; i++) {
    FlashAttentionBwd<HEAD_DIM>::run(q_d, dO_d, k_d, v_d, dq_d, dk_d, dv_d, l_d,
                                     d_d, batch_size, num_q_heads, num_kv_heads,
                                     seq_len, 0, num_sms);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);

  double avg_time = total_time / num_iters;
  double flops =
      (double)batch_size * num_q_heads * seq_len * seq_len * HEAD_DIM * 6.0;

  float tflops = flops / (avg_time * 1e9);

  std::cout << "Batch Size: " << batch_size << std::endl;
  std::cout << "Sequence Length: " << seq_len << std::endl;
  std::cout << "Head Dimension: " << HEAD_DIM << std::endl;
  std::cout << "Average Time: " << avg_time << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // Cleanup
  cudaFree(q_d);
  cudaFree(k_d);
  cudaFree(v_d);
  cudaFree(dO_d);
  cudaFree(dq_d);
  cudaFree(dk_d);
  cudaFree(dv_d);
  cudaFree(l_d);
  cudaFree(d_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  // Test different configurations
  std::vector<int> batch_sizes = {1, 2, 4, 8};
  std::vector<int> seq_lens = {128, 256, 512, 1024, 2048};

  for (int batch_size : batch_sizes) {
    for (int seq_len : seq_lens) {
      std::cout << "\n=== Testing Configuration ===" << std::endl;
      benchmark_flash_attn_bwd<64>(batch_size, 32, 32, seq_len);
    }
  }

  return 0;
}
