import numpy as np
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import mxfp8_quantize


def torch_mxfp8_quantize(
    V: torch.Tensor # (M, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    # Function is naive for clarity, should not be like this in production
    assert len(V.shape) == 2
    assert V.shape[0] % 128 == 0
    assert V.shape[1] % 128 == 0

    # Following not the OCP MX specs, but the NVIDIA recipe: https://arxiv.org/pdf/2506.08027
    # This specifically follows the Appendix (page 13)
    # To be more precise, we have to use Blackwell hardware (thus, in the kernel)
    block_amax = torch.amax(torch.abs(V).view(V.shape[0], V.shape[1] // 32, 32), dim=-1)
    dest_max = 448.0
    decode_scale = block_amax / dest_max
    V_sc_unswizzled = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=-127)
    V_fp8 = (V / (2 ** V_sc_unswizzled.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)

    # Torch (up to 2.8) does not support float8_e8m0, so we need to manually convert to uint8
    fp8e8m0_bias = 127
    V_sc_unswizzled = (V_sc_unswizzled + fp8e8m0_bias).to(torch.uint8)

    return (
        V_fp8,          # (M, N)
        V_sc_unswizzled # (M, N // 32)
    )


def torch_mxfp8_dequantize(
    V_fp8: torch.Tensor,          # (M, N)
    V_sc_unswizzled: torch.Tensor # (M, N // 32)
) -> torch.Tensor:
    # Function is naive for clarity, should not be like this in production
    assert len(V_fp8.shape) == 2
    assert len(V_sc_unswizzled.shape) == 2
    assert V_fp8.dtype == torch.float8_e4m3fn
    assert V_sc_unswizzled.dtype == torch.uint8
    assert V_fp8.shape[0] == V_sc_unswizzled.shape[0]
    assert V_fp8.shape[1] == V_sc_unswizzled.shape[1] * 32

    # Torch (up to 2.8) does not support float8_e8m0, so we need to manually convert
    fp8e8m0_bias = 127
    scale = 2 ** (V_sc_unswizzled.to(torch.float32) - fp8e8m0_bias)

    return V_fp8.to(torch.float32) * scale.repeat_interleave(32, dim=-1)


def scale_swizzle(
    V_sc_unswizzled: torch.Tensor # (M, N // 32)
) -> torch.Tensor:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    assert len(V_sc_unswizzled.shape) == 2
    assert V_sc_unswizzled.dtype == torch.uint8
    assert V_sc_unswizzled.shape[0] % 128 == 0
    assert (V_sc_unswizzled.shape[1] * 32) % 128 == 0

    M_BLOCK = 128
    N_BLOCK = 4 # 128 / 32

    M, N_32 = V_sc_unswizzled.shape

    V_sc = V_sc_unswizzled        # (M, N_32)
    V_sc = V_sc.reshape(          # (M / 128, 128, N_32 / 4, 4)
        M // M_BLOCK, M_BLOCK,
        N_32 // N_BLOCK, N_BLOCK
    )
    V_sc = V_sc.transpose(1, 2)   # (M / 128, N_32 / 4, 128, 4) --> last 2 dims are all we need per MM
    V_sc = V_sc.reshape(          # (M / 128, N_32 / 4, 4, 32, 4)
        M // M_BLOCK, N_32 // N_BLOCK,
        4, M_BLOCK // 4, N_BLOCK
    )
    V_sc = V_sc.transpose(-2, -3) # (M / 128, N_32 / 4, 32, 4, 4)
    V_sc = V_sc.reshape(          # (M / 128, N_32 / 4, 32, 16)
        M // M_BLOCK,
        N_32 // N_BLOCK,
        M_BLOCK // 4, N_BLOCK * 4
    )

    return V_sc.contiguous()


def check_diff(
    name: str,
    A: torch.Tensor,
    A_ref: torch.Tensor
) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    print(f"===============================================================================")
    print(f"<{name}>")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


if __name__ == '__main__':
    # Matrix dimensions
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 204800
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    print(f"{M=}, {N=}")

    # Generate reference outputs and input matrix
    A_fp8_ref = ((torch.rand(M, N, dtype=torch.float32, device="cuda") * 2 - 1) * 448).to(torch.float8_e4m3fn)
    A_sc_unswizzled_ref = torch.randint(127 - 20, 127 + 20, (M, N // 32), dtype=torch.uint8, device="cuda")
    A_sc_ref = scale_swizzle(A_sc_unswizzled_ref)
    A_bf16 = torch_mxfp8_dequantize(A_fp8_ref, A_sc_unswizzled_ref).to(torch.bfloat16)

    # Run PyTorch version and check correctness
    A_fp8_torch, A_sc_unswizzled_torch = torch_mxfp8_quantize(A_bf16)
    A_sc_torch = scale_swizzle(A_sc_unswizzled_torch)
    check_diff("Torch-FP8", A_fp8_torch, A_fp8_ref)
    check_diff("Torch-SC", A_sc_torch, A_sc_ref)

    # Run our version and check correctness
    A_fp8_tk = torch.zeros_like(A_fp8_ref)
    A_sc_tk = torch.zeros_like(A_sc_ref)
    mxfp8_quantize(A_bf16, A_fp8_tk, A_sc_tk)
    torch.cuda.synchronize()
    check_diff("TK-FP8", A_fp8_tk, A_fp8_ref)
    check_diff("TK-SC", A_sc_tk, A_sc_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

    for i in range(NUM_WARMUPS):
        mxfp8_quantize(A_bf16, A_fp8_tk, A_sc_tk)

    l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
    l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
    cache_clear = lambda: l2_cache.random_(0, 1)

    for i in range(NUM_ITERS):
        cache_clear()
        start_events[i].record()
        mxfp8_quantize(A_bf16, A_fp8_tk, A_sc_tk)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    gb = M * N * (2 + 1 + 1 / 32) * 1e-9
    gbps = gb / avg_time

    print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
    print(f"Average throughput: {gbps:.2f} GB/s")
