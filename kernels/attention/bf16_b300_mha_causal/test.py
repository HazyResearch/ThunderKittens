#!/usr/bin/env python3
"""
Correctness + benchmark for bf16_b300_mha_causal kernel.

Correctness: Flash Attention as reference.
Benchmarking methodology (https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2):
  - Bitwise-identical inputs via splitmix64 hash
  - Uniform(-1, 1) bf16 initialization with deterministic seeds
  - L2 cache eviction via multiple input buffer groups (3x L2 size)
  - 500 warmup iterations, 100 profiling iterations
  - Single CUDA event pair around the full benchmark loop
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

import _C

SEED = 2024


def splitmix_bf16(count: int, seed: int, min_val: float, max_val: float) -> torch.Tensor:
    """Splitmix64 hash -> uniform bf16, bitwise identical to common.cuh fill_kernel."""
    idx = np.arange(count, dtype=np.uint64)
    x = np.uint64(seed) + idx
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    vals = u * np.float32(max_val - min_val) + np.float32(min_val)
    return torch.from_numpy(vals).to(torch.bfloat16).cuda()


def flops_fwd(B: int, S: int, H: int, Dqk: int, Dv: int, causal: bool = False) -> float:
    f = 2.0 * B * H * S * S * (Dqk + Dv)
    return f * 0.5 if causal else f


def create_inputs(B, S, H, Dqk, Dv, seed):
    """Create a single set of (Q, K, V, O, LSE) tensors."""
    count_qk = B * S * H * Dqk
    count_v = B * S * H * Dv
    Q = splitmix_bf16(count_qk, seed, -1.0, 1.0).view(B, S, H, Dqk)
    K = splitmix_bf16(count_qk, seed + 1, -1.0, 1.0).view(B, S, H, Dqk)
    V = splitmix_bf16(count_v, seed + 2, -1.0, 1.0).view(B, S, H, Dv)
    O = torch.zeros(B, S, H, Dv, dtype=torch.bfloat16, device="cuda")
    LSE = torch.zeros(B, H, 1, S, dtype=torch.float32, device="cuda")
    return Q, K, V, O, LSE


def _fwd(S):
    """Select persistent kernel for short sequences, non-persistent otherwise."""
    return _C.forward_persistent if S <= 4096 else _C.forward


def check_correctness(B, S, H, Dqk, Dv):
    """Check kernel output against Flash Attention reference."""
    Q, K, V, O, LSE = create_inputs(B, S, H, Dqk, Dv, SEED)

    _fwd(S)(Q, K, V, O, LSE)
    torch.cuda.synchronize()

    # Flash Attention reference - pad V to match Dqk since FA2 needs equal head dims
    softmax_scale = 1.0 / math.sqrt(Dqk)
    V_pad = F.pad(V, (0, Dqk - Dv))
    O_ref = flash_attn_func(Q, K, V_pad, softmax_scale=softmax_scale, causal=True)
    O_ref = O_ref[..., :Dv]

    # Compare
    O_f = O.float()
    O_ref_f = O_ref.float()
    diff = (O_f - O_ref_f).abs()
    print(f"abs mean: {O_f.abs().mean().item():12g}")
    print(f"abs max:  {O_f.abs().max().item():12g}")
    print(f"err mean: {diff.mean().item():12g}")
    print(f"err max:  {diff.max().item():12g}")


@torch.inference_mode()
def bench_shape(B, S, H, Dqk, Dv, warmup=500, iters=100):
    time.sleep(0.5)  # cooldown between shapes

    # L2 cache eviction - multiple buffer groups
    l2_cache_size = torch.cuda.get_device_properties(0).L2_cache_size
    arg_size = (2 * B * S * H * Dqk + B * S * H * Dv) * 2  # bytes
    ideal_arg_size = l2_cache_size * 3
    arg_group_count = 1 if arg_size > ideal_arg_size else (ideal_arg_size // arg_size) + 1

    # Create input groups
    groups = []
    for i in range(arg_group_count):
        groups.append(create_inputs(B, S, H, Dqk, Dv, SEED + i * 100))

    # Warmup
    for i in range(warmup):
        Q, K, V, O, LSE = groups[i % arg_group_count]
        _fwd(S)(Q, K, V, O, LSE)

    # Benchmark - single CUDA event pair
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        Q, K, V, O, LSE = groups[i % arg_group_count]
        _fwd(S)(Q, K, V, O, LSE)
    end.record()
    end.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / iters
    tflops = flops_fwd(B, S, H, Dqk, Dv, causal=True) / (avg_us * 1e-6) / 1e12

    return avg_us, tflops


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, " cuda:", torch.version.cuda)

    B, H = 16, 16
    Dqk, Dv = 192, 128
    seqs = [2048, 4096, 8192, 16384]

    for S in seqs:
        header = f"batch={B} seq_len={S} num_heads={H} Dqk={Dqk} Dvo={Dv}"
        print(f"\n{'':->20}  {header}  {'':->20}")
        print("Checking correctness")
        check_correctness(B, S, H, Dqk, Dv)
        print("Running benchmark")
        avg_us, tflops = bench_shape(B, S, H, Dqk, Dv)
        print(f"Average kernel execution time: {avg_us:.2f} us")
        print(f"Achieved performance: {tflops:.2f} TFLOPs")


if __name__ == "__main__":
    main()
