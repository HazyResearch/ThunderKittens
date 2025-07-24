import torch

import random
import torch
from typing import List, Tuple

import thunderkittens
import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from deep_gemm.jit_kernels.utils import get_m_alignment_for_contiguous_layout

torch.manual_seed(42)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct_contiguous_grouped(num_groups: int, expected_m_per_group: int, k: int, n: int, bf16_input=False) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment = get_m_alignment_for_contiguous_layout()
    # group_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    group_ms = [expected_m_per_group  for _ in range(num_groups)]
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)    
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = i
        ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end
    ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    if bf16_input:
        return m, x, y, m_indices, out, ref_out
    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    return m, x_fp8, y_fp8, m_indices, out, ref_out


def test_m_grouped_gemm_contiguous_deepgemm() -> None:
    print('Testing grouped contiguous GEMM for deepgemm:')

    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                   (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                   (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        # NOTES: we should mask the unfilled part before calculating difference
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ({num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked_deepgemm() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            for i in range(10):
                x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, 4096, expected_m_per_group, k, n)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)

            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_contiguous_tk() -> None:
    print('Testing grouped contiguous GEMM for tk:')

    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                   (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                   (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        # NOTES: we should mask the unfilled part before calculating difference
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)

        ht_out = torch.empty_like(out)
        thunderkittens.group_gemm(
            x_fp8[0], # x_fp8 is a tuple (x, x_inv_s)
            y_fp8[0], # y_fp8 is a tuple (y, y_inv_s)
            x_fp8[1].transpose(0, 1).contiguous(), # x_inv_s
            y_fp8[1],
            m_indices,
            ht_out)
        ht_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), ht_out)
        diff = calc_diff(ht_out, ref_out)
        assert diff < 0.001, f'TK has big difference with float: {m=}, {k=}, {n=}, {diff:.5f}'
        assert torch.allclose(out, ht_out.bfloat16(), atol=1e-3, rtol=1e-2), f'TK has big difference with deepgemm: {m=}, {k=}, {n=}'

        # tk scale shape need to b (k, m)
        x_fp8 = (x_fp8[0], x_fp8[1].transpose(0, 1).contiguous())
        def test_func():
            thunderkittens.group_gemm(
                x_fp8[0], # x_fp8 is a tuple (x, x_inv_s)
                y_fp8[0], # y_fp8 is a tuple (y, y_inv_s)
                x_fp8[1], # x_inv_s
                y_fp8[1],
                m_indices,
                ht_out)

        t = bench_kineto(test_func, 'matmul_template', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ({num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
        
    print()


if __name__ == '__main__':
    test_m_grouped_gemm_contiguous_deepgemm()
    test_m_grouped_gemm_contiguous_tk()
