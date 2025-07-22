#include "h100_lcf_bwd.cuh"

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

// #include <vector> // Removed: replaced by thrust::device_vector

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

// 在 flash_attention_bwd_reference 旁边添加这个新函数

/**
 * @brief FlashAttention forward pass的引用实现。
 *
 * 该函数计算标准的缩放点积注意力，并返回输出O和log-sum-exp (LSE)统计量。
 * LSE对于验证FlashAttention的反向传播至关重要。
 *
 * @param q 输入Query张量，形状为 (B, H, N_q, D)。
 * @param k 输入Key张量，形状为 (B, H, N_kv, D)。
 * @param v 输入Value张量，形状为 (B, H, N_kv, D)。
 * @param is_causal 是否应用因果掩码。
 * @return 一个包含输出O和LSE张量的元组。
 */
// std::tuple<at::Tensor, at::Tensor> flash_attention_fwd_reference(
//     const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
//     bool is_causal) {
//   // 确保输入是浮点类型
//   TORCH_CHECK(q.is_floating_point(), "Q must be a floating point tensor");
//   TORCH_CHECK(k.is_floating_point(), "K must be a floating point tensor");
//   TORCH_CHECK(v.is_floating_point(), "V must be a floating point tensor");

//   // 获取维度信息
//   const auto head_dim = q.size(-1);
//   const auto seq_len_q = q.size(-2);
//   const auto seq_len_kv = k.size(-2);
//   const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

//   // S = (Q @ K.T) * scale
//   at::Tensor s = torch::matmul(q, k.transpose(-2, -1)) * scale;

//   // 如果是因果注意力，应用掩码
//   if (is_causal) {
//     TORCH_CHECK(
//         seq_len_q == seq_len_kv,
//         "Causal attention requires Q and K to have the same sequence
//         length.");
//     auto mask =
//         torch::ones({seq_len_q, seq_len_q}, q.options().dtype(torch::kBool))
//             .triu_(1);
//     s.masked_fill_(mask, -std::numeric_limits<float>::infinity());
//   }

//   // 计算 LSE (log-sum-exp)
//   // LSE = log(sum(exp(S)))
//   // 为了数值稳定性，使用 log-softmax 的技巧: LSE = S_max + log(sum(exp(S -
//   // S_max)))
//   at::Tensor s_max = std::get<0>(torch::max(s, -1, /*keepdim=*/true));
//   // 处理-inf的情况，当一行全是-inf时，s_max也是-inf，exp(s-s_max)会是NaN
//   s_max.masked_fill_(s_max == -std::numeric_limits<float>::infinity(), 0.0);
//   at::Tensor exp_s = torch::exp(s - s_max);
//   at::Tensor sum_exp_s = torch::sum(exp_s, -1, /*keepdim=*/true);
//   at::Tensor lse = s_max + torch::log(sum_exp_s);

//   // P = softmax(S) = exp(S - LSE)
//   at::Tensor p = torch::exp(s - lse);
//   p = p.to(q.dtype());

//   // O = P @ V
//   at::Tensor o = torch::matmul(p, v);

//   return std::make_tuple(o, lse);
// }

std::tuple<at::Tensor, at::Tensor> flash_attention_fwd_reference(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    bool is_causal) {
  // 确保输入是浮点类型
  TORCH_CHECK(q.is_floating_point(), "Q must be a floating point tensor");
  TORCH_CHECK(k.is_floating_point(), "K must be a floating point tensor");
  TORCH_CHECK(v.is_floating_point(), "V must be a floating point tensor");

  // 确保输入是4维张量 (B, H, N, D)
  TORCH_CHECK(q.dim() == 4, "Q must be a 4D tensor");
  TORCH_CHECK(k.dim() == 4, "K must be a 4D tensor");
  TORCH_CHECK(v.dim() == 4, "V must be a 4D tensor");

  // 获取维度信息
  const auto batch_size = q.size(0);
  const auto num_heads_q = q.size(1);
  const auto seq_len_q = q.size(2);
  const auto head_dim = q.size(3);

  const auto num_heads_kv = k.size(1);
  const auto seq_len_kv = k.size(2);

  // 检查 GQA 的约束
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  const int num_groups = num_heads_q / num_heads_kv;

  // 准备 K 和 V 以进行 Grouped-Query Attention
  at::Tensor k_grouped, v_grouped;

  if (num_groups == 1) {
    // 标准的多头注意力 (MHA)，无需操作
    k_grouped = k;
    v_grouped = v;
  } else {
    // GQA/MQA: 需要将 K/V 的头重复以匹配 Q 的头
    // k: (B, H_KV, N, D) -> (B, H_KV, 1, N, D)
    //      -> expand to (B, H_KV, num_groups, N, D)
    //      -> view as (B, H_Q, N, D)
    k_grouped = k.unsqueeze(2)
                    .expand({batch_size, num_heads_kv, num_groups, seq_len_kv,
                             head_dim})
                    .reshape({batch_size, num_heads_q, seq_len_kv, head_dim});
    v_grouped = v.unsqueeze(2)
                    .expand({batch_size, num_heads_kv, num_groups, seq_len_kv,
                             head_dim})
                    .reshape({batch_size, num_heads_q, seq_len_kv, head_dim});
  }

  // 使用 float32 进行中间计算以保证精度，这与 CUDA kernel 的做法一致
  at::Tensor q_f32 = q.to(torch::kFloat32);
  at::Tensor k_f32 = k_grouped.to(torch::kFloat32);
  at::Tensor v_f32 = v_grouped.to(torch::kFloat32);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // S = (Q @ K.T) * scale
  // q_f32: (B, H_Q, N_q, D)
  // k_f32.transpose: (B, H_Q, D, N_kv)
  // s: (B, H_Q, N_q, N_kv)
  at::Tensor s = torch::matmul(q_f32, k_f32.transpose(-2, -1)) * scale;

  // 如果是因果注意力，应用掩码
  if (is_causal) {
    TORCH_CHECK(
        seq_len_q == seq_len_kv,
        "Causal attention requires Q and K to have the same sequence length.");
    auto mask =
        torch::ones({seq_len_q, seq_len_q}, q.options().dtype(torch::kBool))
            .triu_(1);
    s.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  }

  // --- LSE (log-sum-exp) 计算，严格匹配 CUDA kernel ---
  // 标准 LSE 计算
  // LSE_std = log(sum(exp(S)))
  // 为了数值稳定性: LSE_std = S_max + log(sum(exp(S - S_max)))
  at::Tensor s_max = std::get<0>(torch::max(s, -1, /*keepdim=*/true));
  // 处理-inf的情况，当一行全是-inf时，s_max也是-inf，exp(s-s_max)会是NaN
  s_max.masked_fill_(s_max == -std::numeric_limits<float>::infinity(), 0.0);
  at::Tensor exp_s = torch::exp(s - s_max);
  at::Tensor sum_exp_s = torch::sum(exp_s, -1, /*keepdim=*/true);
  at::Tensor lse_std = s_max + torch::log(sum_exp_s);

  // --- 关键修改：应用 CUDA kernel 中的特殊缩放和取反 ---
  // CUDA kernel 计算的是: L_final = -sqrt(D) * LSE_std
  // const float final_lse_scale = -std::sqrt(static_cast<float>(head_dim));
  // at::Tensor lse_final = lse_std * final_lse_scale;

  // P = softmax(S) = exp(S - LSE_std)
  // 注意：这里的 softmax 仍然使用标准的 LSE (lse_std)
  at::Tensor p = torch::exp(s - lse_std);

  // O = P @ V
  at::Tensor o = torch::matmul(p, v_f32);

  // 将结果转换回原始数据类型并返回
  return std::make_tuple(o.to(q.dtype()), lse_std.to(q.dtype()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_bwd_reference(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& dO, bool is_causal) {
  // 确保输入是浮点类型
  TORCH_CHECK(q.is_floating_point(), "Q must be a floating point tensor");
  TORCH_CHECK(k.is_floating_point(), "K must be a floating point tensor");
  TORCH_CHECK(v.is_floating_point(), "V must be a floating point tensor");
  TORCH_CHECK(dO.is_floating_point(), "dO must be a floating point tensor");

  // 获取维度信息
  const auto head_dim = q.size(-1);
  const auto seq_len_q = q.size(-2);
  const auto seq_len_kv = k.size(-2);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // --- 步骤 1: 重新计算前向传播中的注意力矩阵 P ---
  // S = (Q @ K.T) * scale
  at::Tensor s = torch::matmul(q, k.transpose(-2, -1)) * scale;

  // 如果是因果注意力，应用掩码
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
  // 将P转换为与Q/K/V相同的类型，以防混合精度
  p = p.to(q.dtype());

  // --- 步骤 2: 计算梯度 ---

  // dV = P.T @ dO
  at::Tensor dV = torch::matmul(p.transpose(-2, -1), dO);

  // dP = dO @ V.T
  at::Tensor dP = torch::matmul(dO, v.transpose(-2, -1));

  // dS = P * (dP - row_sum(P * dP))
  // 这是softmax反向传播的关键步骤
  at::Tensor p_x_dP = p * dP;  // 逐元素相乘
  at::Tensor row_sum = torch::sum(p_x_dP, -1, /*keepdim=*/true);
  at::Tensor dS = p * (dP - row_sum);

  // 撤销缩放
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

  // For simplicity, let's assume tile_h_qo and tile_h are fixed for now
  // From ring_attention_bwd.cuh:
  // static constexpr int tile_h_qo = 64; // Q/O tile height
  // static constexpr int tile_h = 64; // K/V tile height
  // The sequence length must be a multiple of tile_h.

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

  // --- 3. 将生成的 O, L, D 从Torch张量复制回Host vectors ---
  // 这是为了将它们传递给你的CUDA内核
  thrust::host_vector<nv_bfloat16> h_o_init(o_size / sizeof(nv_bfloat16));
  thrust::host_vector<float> h_l_init(l_size / sizeof(float));
  thrust::host_vector<float> h_d_init(d_size / sizeof(float));

  auto o_cpu_out = o_tensor.to(torch::kCPU).contiguous();
  auto l_cpu_out = l_tensor.to(torch::kCPU).contiguous();
  auto d_cpu_out = d_tensor.to(torch::kCPU).contiguous();
  auto dO_cpu_out =
      dO_tensor.to(torch::kCPU).contiguous();  // 我们也需要dO的host copy

  memcpy(h_o_init.data(), o_cpu_out.data_ptr(), o_size);
  memcpy(h_l_init.data(), l_cpu_out.data_ptr(), l_size);
  memcpy(h_d_init.data(), d_cpu_out.data_ptr(), d_size);

  thrust::host_vector<nv_bfloat16> h_dO_init(o_size / sizeof(nv_bfloat16));
  memcpy(h_dO_init.data(), dO_cpu_out.data_ptr(), o_size);

  // --- 4. 设置设备内存并启动你的CUDA内核 ---
  std::cout << "Setting up device memory for custom kernel..." << std::endl;

  // Device memory allocation using thrust::device_vector
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

  std::cout << "Launching RingAttentionBwd kernel..." << std::endl;
  // Launch the kernel
  // The template parameter is `head_dim`. The `is_causal` parameter is the
  // second template parameter to `attn_bwd_template`. We are using
  // `RingAttentionBwd` directly now, which takes `head_dim` as template
  // parameter. Inside `RingAttentionBwd::run`, it constructs
  // `attn_bwd_template<head_dim, false>`, so it's non-causal by default.
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

  // 确保所有数据都在同一个设备上
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

  // 计算最大绝对误差和平均绝对误差
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
