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

    # Group size
    l2_size = 128 * 1024 * 1024
    size_per_group = M * N * 2
    num_groups = (l2_size // size_per_group + 1) * 100
    print(f"{M=}, {N=}, {num_groups=}")

    # Generate reference outputs and input matrix
    groups = []
    for i in range(num_groups):
        A_fp8_ref = ((torch.rand(M, N, dtype=torch.float32, device="cuda") * 2 - 1) * 448).to(torch.float8_e4m3fn)
        A_sc_unswizzled_ref = torch.randint(127 - 20, 127 + 20, (M, N // 32), dtype=torch.uint8, device="cuda")
        A_sc_ref = scale_swizzle(A_sc_unswizzled_ref)
        A_bf16 = torch_mxfp8_dequantize(A_fp8_ref, A_sc_unswizzled_ref).to(torch.bfloat16)
        A_fp8 = torch.zeros_like(A_fp8_ref)
        A_sc = torch.zeros_like(A_sc_ref)
        groups.append((A_bf16, A_fp8, A_sc))

    # Run PyTorch version and check correctness using the last input
    A_fp8_torch, A_sc_unswizzled_torch = torch_mxfp8_quantize(A_bf16)
    A_sc_torch = scale_swizzle(A_sc_unswizzled_torch)
    check_diff("Torch-FP8", A_fp8_torch, A_fp8_ref)
    check_diff("Torch-SC", A_sc_torch, A_sc_ref)

    # Run our version and check correctness using the last input
    mxfp8_quantize(A_bf16, A_fp8, A_sc)
    torch.cuda.synchronize()
    check_diff("TK-FP8", A_fp8, A_fp8_ref)
    check_diff("TK-SC", A_sc, A_sc_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        mxfp8_quantize(*groups[i % num_groups])
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        mxfp8_quantize(*groups[i % num_groups])
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    gb = M * N * (2 + 1 + 1 / 32) * 1e-9
    gbps = gb / avg_time

    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average throughput: {gbps:.2f} GB/s")
