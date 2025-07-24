#include "h100_lcsf_bwd.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>  // Required for std::sqrt
#include <iostream>
#include <limits>  // Required for std::numeric_limits
#include <random>

// Helper to check CUDA errors
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                    \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

float truncated_normal(std::mt19937& gen,
                       std::normal_distribution<float>& dist) {
  float value;
  do {
    value = dist(gen);
  } while (value <= -1.0f || value >= 1.0f);
  return value;
}

std::tuple<at::Tensor, at::Tensor> flash_attention_fwd_reference(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    bool is_causal) {
  TORCH_CHECK(q.is_floating_point(), "Q must be a floating point tensor");
  TORCH_CHECK(k.is_floating_point(), "K must be a floating point tensor");
  TORCH_CHECK(v.is_floating_point(), "V must be a floating point tensor");

  TORCH_CHECK(q.dim() == 4, "Q must be a 4D tensor");
  TORCH_CHECK(k.dim() == 4, "K must be a 4D tensor");
  TORCH_CHECK(v.dim() == 4, "V must be a 4D tensor");

  const auto batch_size = q.size(0);
  const auto num_heads_q = q.size(1);
  const auto seq_len_q = q.size(2);
  const auto head_dim = q.size(3);

  const auto num_heads_kv = k.size(1);
  const auto seq_len_kv = k.size(2);

  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  const int num_groups = num_heads_q / num_heads_kv;

  at::Tensor k_grouped, v_grouped;

  if (num_groups == 1) {
    k_grouped = k;
    v_grouped = v;
  } else {
    k_grouped = k.unsqueeze(2)
                    .expand({batch_size, num_heads_kv, num_groups, seq_len_kv,
                             head_dim})
                    .reshape({batch_size, num_heads_q, seq_len_kv, head_dim});
    v_grouped = v.unsqueeze(2)
                    .expand({batch_size, num_heads_kv, num_groups, seq_len_kv,
                             head_dim})
                    .reshape({batch_size, num_heads_q, seq_len_kv, head_dim});
  }

  at::Tensor q_f32 = q.to(torch::kFloat32);
  at::Tensor k_f32 = k_grouped.to(torch::kFloat32);
  at::Tensor v_f32 = v_grouped.to(torch::kFloat32);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // S = (Q @ K.T) * scale
  // q_f32: (B, H_Q, N_q, D)
  // k_f32.transpose: (B, H_Q, D, N_kv)
  // s: (B, H_Q, N_q, N_kv)
  at::Tensor s = torch::matmul(q_f32, k_f32.transpose(-2, -1)) * scale;

  if (is_causal) {
    TORCH_CHECK(
        seq_len_q == seq_len_kv,
        "Causal attention requires Q and K to have the same sequence length.");
    auto mask =
        torch::ones({seq_len_q, seq_len_q}, q.options().dtype(torch::kBool))
            .triu_(1);
    s.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  }

  at::Tensor s_max = std::get<0>(torch::max(s, -1, /*keepdim=*/true));
  s_max.masked_fill_(s_max == -std::numeric_limits<float>::infinity(), 0.0);
  at::Tensor exp_s = torch::exp(s - s_max);
  at::Tensor sum_exp_s = torch::sum(exp_s, -1, /*keepdim=*/true);
  at::Tensor lse_std = s_max + torch::log(sum_exp_s);

  at::Tensor p = torch::exp(s - lse_std);

  // O = P @ V
  at::Tensor o = torch::matmul(p, v_f32);

  return std::make_tuple(o.to(q.dtype()), lse_std.to(q.dtype()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_bwd_reference(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& dO, bool is_causal) {
  TORCH_CHECK(q.is_floating_point(), "Q must be a floating point tensor");
  TORCH_CHECK(k.is_floating_point(), "K must be a floating point tensor");
  TORCH_CHECK(v.is_floating_point(), "V must be a floating point tensor");
  TORCH_CHECK(dO.is_floating_point(), "dO must be a floating point tensor");

  const auto head_dim = q.size(-1);
  const auto seq_len_q = q.size(-2);
  const auto seq_len_kv = k.size(-2);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // S = (Q @ K.T) * scale
  at::Tensor s = torch::matmul(q, k.transpose(-2, -1)) * scale;

  if (is_causal) {
    TORCH_CHECK(
        seq_len_q == seq_len_kv,
        "Causal attention requires Q and K to have the same sequence length.");
    auto mask =
        torch::ones({seq_len_q, seq_len_q}, q.options().dtype(torch::kBool))
            .triu_(1);
    s.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  }

  // P = softmax(S)
  at::Tensor p = torch::softmax(s, -1);
  p = p.to(q.dtype());

  // dV = P.T @ dO
  at::Tensor dV = torch::matmul(p.transpose(-2, -1), dO);

  // dP = dO @ V.T
  at::Tensor dP = torch::matmul(dO, v.transpose(-2, -1));

  // dS = P * (dP - row_sum(P * dP))
  at::Tensor p_x_dP = p * dP;
  at::Tensor row_sum = torch::sum(p_x_dP, -1, /*keepdim=*/true);
  at::Tensor dS = p * (dP - row_sum);

  dS = dS * scale;

  // dQ = dS @ K
  at::Tensor dQ = torch::matmul(dS, k);

  // dK = dS.T @ Q
  at::Tensor dK = torch::matmul(dS.transpose(-2, -1), q);

  return std::make_tuple(dQ, dK, dV);
}

int main() {
  // Define problem dimensions
  const uint32_t batch_size = 1;
  const uint32_t num_q_heads = 1;
  const uint32_t num_kv_heads = 1;
  const uint32_t seq_len = 256;  // Must be a multiple of tile_h
  const uint32_t head_dim = 64;  // Must be 64 or 128

  // Calculate tensor sizes
  size_t q_size =
      batch_size * num_q_heads * seq_len * head_dim * sizeof(nv_bfloat16);
  size_t o_size =
      batch_size * num_q_heads * seq_len * head_dim * sizeof(nv_bfloat16);
  size_t k_size =
      batch_size * num_kv_heads * seq_len * head_dim * sizeof(nv_bfloat16);
  size_t v_size =
      batch_size * num_kv_heads * seq_len * head_dim * sizeof(nv_bfloat16);
  size_t dq_size =
      batch_size * num_q_heads * seq_len * head_dim * sizeof(float);
  size_t dk_size =
      batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);
  size_t dv_size =
      batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);
  size_t l_size =
      batch_size * num_q_heads * seq_len * sizeof(float);  // Logits L
  size_t d_size =
      batch_size * num_q_heads * seq_len * sizeof(float);  // Denominator D

  // Host memory allocation (using thrust::host_vector for initial data if
  // needed, or initialize directly on device) For simplicity, we'll directly
  // use thrust::device_vector and initialize on device or copy from host.

  // Initialize host data (simple example values for copying to device)
  // In a real test, these would come from a forward pass or specific test
  // cases.
  thrust::host_vector<nv_bfloat16> h_q_init(q_size / sizeof(nv_bfloat16));
  thrust::host_vector<nv_bfloat16> h_k_init(k_size / sizeof(nv_bfloat16));
  thrust::host_vector<nv_bfloat16> h_v_init(v_size / sizeof(nv_bfloat16));

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> normal{0.0f, 1.0f};

  for (size_t i = 0; i < h_q_init.size(); ++i)
    h_q_init[i] = static_cast<nv_bfloat16>(truncated_normal(gen, normal));
  for (size_t i = 0; i < h_k_init.size(); ++i)
    h_k_init[i] = static_cast<nv_bfloat16>(truncated_normal(gen, normal));
  for (size_t i = 0; i < h_v_init.size(); ++i)
    h_v_init[i] = static_cast<nv_bfloat16>(truncated_normal(gen, normal));

  std::cout << "Generating reference O, L, D using fwd pass..." << std::endl;

  auto options_bf16_cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  auto q_cpu = torch::from_blob(h_q_init.data(),
                                {batch_size, num_q_heads, seq_len, head_dim},
                                options_bf16_cpu);
  auto k_cpu = torch::from_blob(h_k_init.data(),
                                {batch_size, num_kv_heads, seq_len, head_dim},
                                options_bf16_cpu);
  auto v_cpu = torch::from_blob(h_v_init.data(),
                                {batch_size, num_kv_heads, seq_len, head_dim},
                                options_bf16_cpu);

  auto q_tensor = q_cpu.to(torch::kCUDA);
  auto k_tensor = k_cpu.to(torch::kCUDA);
  auto v_tensor = v_cpu.to(torch::kCUDA);

  auto [o_tensor, l_tensor] =
      flash_attention_fwd_reference(q_tensor, k_tensor, v_tensor, false);

  auto dO_tensor = torch::randn_like(o_tensor);

  auto d_tensor = torch::sum(dO_tensor * o_tensor, -1, /*keepdim=*/true);

  thrust::host_vector<nv_bfloat16> h_o_init(o_size / sizeof(nv_bfloat16));
  thrust::host_vector<float> h_l_init(l_size / sizeof(float));
  thrust::host_vector<float> h_d_init(d_size / sizeof(float));

  auto o_cpu_out = o_tensor.to(torch::kCPU).contiguous();
  auto l_cpu_out = l_tensor.to(torch::kCPU).contiguous();
  auto d_cpu_out = d_tensor.to(torch::kCPU).contiguous();
  auto dO_cpu_out = dO_tensor.to(torch::kCPU).contiguous();

  memcpy(h_o_init.data(), o_cpu_out.data_ptr(), o_size);
  memcpy(h_l_init.data(), l_cpu_out.data_ptr(), l_size);
  memcpy(h_d_init.data(), d_cpu_out.data_ptr(), d_size);

  thrust::host_vector<nv_bfloat16> h_dO_init(o_size / sizeof(nv_bfloat16));
  memcpy(h_dO_init.data(), dO_cpu_out.data_ptr(), o_size);

  std::cout << "Setting up device memory for custom kernel..." << std::endl;

  // Device memory allocation using thrust::device_vector
  thrust::device_vector<nv_bfloat16> d_q(h_q_init);
  thrust::device_vector<nv_bfloat16> d_k(h_k_init);
  thrust::device_vector<nv_bfloat16> d_v(h_v_init);
  thrust::device_vector<nv_bfloat16> d_o(h_o_init);
  thrust::device_vector<nv_bfloat16> d_dO(h_dO_init);  // 使用 dO 而不是 O
  thrust::device_vector<float> d_l(h_l_init);
  thrust::device_vector<float> d_d(h_d_init);

  thrust::device_vector<float> d_dq(dq_size / sizeof(float), 0.0f);
  thrust::device_vector<float> d_dk(dk_size / sizeof(float), 0.0f);
  thrust::device_vector<float> d_dv(dv_size / sizeof(float), 0.0f);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Get number of SMs for kernel launch configuration
  int num_sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

  std::cout << "Launching FlashAttentionBwd kernel..." << std::endl;
  // Launch the kernel
  FlashAttentionBwd<head_dim>::run(thrust::raw_pointer_cast(d_q.data()),
                                   thrust::raw_pointer_cast(d_dO.data()),
                                   thrust::raw_pointer_cast(d_k.data()),
                                   thrust::raw_pointer_cast(d_v.data()),
                                   thrust::raw_pointer_cast(d_dq.data()),
                                   thrust::raw_pointer_cast(d_dk.data()),
                                   thrust::raw_pointer_cast(d_dv.data()),
                                   thrust::raw_pointer_cast(d_l.data()),
                                   thrust::raw_pointer_cast(d_d.data()),
                                   batch_size, num_q_heads, num_kv_heads,
                                   seq_len, stream, num_sms);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "Kernel launch complete." << std::endl;

  // Copy results back to host for verification
  thrust::host_vector<float> h_dq_result = d_dq;
  thrust::host_vector<float> h_dk_result = d_dk;
  thrust::host_vector<float> h_dv_result = d_dv;

  // Run reference implementation
  c10::cuda::CUDAGuard device_guard(q_tensor.device());

  auto [dQ_ref, dK_ref, dV_ref] = flash_attention_bwd_reference(
      q_tensor, k_tensor, v_tensor, dO_tensor, false);

  // Compare results
  std::cout << "\n=== Comparing with FlashAttention Reference ===" << std::endl;

  // Convert reference results back to CPU
  auto dQ_ref_cpu = dQ_ref.cpu().contiguous().to(torch::kFloat32);
  auto dK_ref_cpu = dK_ref.cpu().contiguous().to(torch::kFloat32);
  auto dV_ref_cpu = dV_ref.cpu().contiguous().to(torch::kFloat32);

  // Calculate max absolute error
  float max_error_q = 0.0f;
  float max_error_k = 0.0f;
  float max_error_v = 0.0f;

  auto dQ_ref_ptr = dQ_ref_cpu.data_ptr<float>();
  auto dK_ref_ptr = dK_ref_cpu.data_ptr<float>();
  auto dV_ref_ptr = dV_ref_cpu.data_ptr<float>();

  double sum_error_q = 0.0;
  double sum_error_k = 0.0;
  double sum_error_v = 0.0;
  for (size_t i = 0; i < h_dq_result.size(); ++i) {
    float diff_q = std::abs(h_dq_result[i] - dQ_ref_ptr[i]);
    max_error_q = std::max(max_error_q, diff_q);
    sum_error_q += diff_q;
  }
  for (size_t i = 0; i < h_dk_result.size(); ++i) {
    float diff_k = std::abs(h_dk_result[i] - dK_ref_ptr[i]);
    max_error_k = std::max(max_error_k, diff_k);
    sum_error_k += diff_k;
  }
  for (size_t i = 0; i < h_dv_result.size(); ++i) {
    float diff_v = std::abs(h_dv_result[i] - dV_ref_ptr[i]);
    max_error_v = std::max(max_error_v, diff_v);
    sum_error_v += diff_v;
  }

  double mean_error_q = sum_error_q / h_dq_result.size();
  double mean_error_k = sum_error_k / h_dk_result.size();
  double mean_error_v = sum_error_v / h_dv_result.size();

  std::cout << "Max absolute errors:" << std::endl;
  std::cout << "dQ: " << max_error_q << std::endl;
  std::cout << "dK: " << max_error_k << std::endl;
  std::cout << "dV: " << max_error_v << std::endl;

  std::cout << "Mean absolute errors:" << std::endl;
  std::cout << "dQ: " << mean_error_q << std::endl;
  std::cout << "dK: " << mean_error_k << std::endl;
  std::cout << "dV: " << mean_error_v << std::endl;

  int print_num = 128;

  // Print first few values for comparison
  std::cout << "\nFirst few values comparison:" << std::endl;
  std::cout << "dQ (ours vs reference):" << std::endl;
  for (int i = 0; i < print_num; ++i) {
    std::cout << h_dq_result[i] << " vs " << dQ_ref_ptr[i];
    if ((i + 1) % 8 == 0)
      std::cout << std::endl;
    else
      std::cout << "    ";
  }
  if (print_num % 8 != 0) std::cout << std::endl;

  std::cout << "\ndK (ours vs reference):" << std::endl;
  for (int i = 0; i < print_num; ++i) {
    std::cout << h_dk_result[i] << " vs " << dK_ref_ptr[i];
    if ((i + 1) % 8 == 0)
      std::cout << std::endl;
    else
      std::cout << "    ";
  }
  if (print_num % 8 != 0) std::cout << std::endl;

  std::cout << "\ndV (ours vs reference):" << std::endl;
  for (int i = 0; i < print_num; ++i) {
    std::cout << h_dv_result[i] << " vs " << dV_ref_ptr[i];
    if ((i + 1) % 8 == 0)
      std::cout << std::endl;
    else
      std::cout << "    ";
  }
  if (print_num % 8 != 0) std::cout << std::endl;

  // TODO: Add more robust verification (e.g., comparing with a reference
  // implementation)

  // Device memory is automatically freed by thrust::device_vector destructors
  CUDA_CHECK(cudaStreamDestroy(stream));

  std::cout << "Test finished." << std::endl;

  return 0;
}
