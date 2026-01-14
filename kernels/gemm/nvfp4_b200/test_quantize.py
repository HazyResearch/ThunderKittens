import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_quantize, fp32_to_fp4x2, fp4x2_to_fp32  # type: ignore


def torch_nvfp4_quantize(
    V: torch.Tensor # (M, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    # Function is naive for clarity, should not be like this in production
    assert len(V.shape) == 2
    assert V.shape[0] % 128 == 0
    assert V.shape[1] % 128 == 0

    # For Torch reference, we convert from float32
    V = V.to(torch.float32)

    # Following the NVIDIA recipe: https://arxiv.org/pdf/2509.25149 (Appendix)
    s_global_enc = 6. * 448. / torch.amax(torch.abs(V), dim=None).clamp(min=1e-12)
    s_global_dec = 1. / s_global_enc
    s_local_dec = torch.amax(torch.abs(V).view(V.shape[0], V.shape[1] // 16, 16), dim=-1) / 6.
    s_local_dec_e4m3 = (s_local_dec * s_global_enc).to(torch.float8_e4m3fn) # round-to-even
    s_local_dec = s_local_dec_e4m3.to(torch.float32) # choked
    s_enc = 1. / (s_local_dec * s_global_dec).clamp(min=1e-12)

    V_fp4x2 = fp32_to_fp4x2(V * s_enc.repeat_interleave(16, dim=-1)) # round-to-even or stochastic (refer to the recipe)
    V_sc_unswizzled = s_local_dec_e4m3 # alias for prettiness
    V_sc_global = s_global_dec.unsqueeze(0)

    return (
        V_fp4x2,         # (M, N // 2)  fp4e2m1x2
        V_sc_unswizzled, # (M, N // 16) fp8e4m3
        V_sc_global      # (1,)         fp32
    )


def torch_nvfp4_dequantize(
    V_fp4x2: torch.Tensor,         # (M, N // 2)  fp4e2m1x2
    V_sc_unswizzled: torch.Tensor, # (M, N // 16) fp8e4m3
    V_sc_global: torch.Tensor      # (1,)         fp32
) -> torch.Tensor:
    # Function is naive for clarity, should not be like this in production
    assert len(V_fp4x2.shape) == 2
    assert len(V_sc_unswizzled.shape) == 2
    assert len(V_sc_global.shape) == 1
    assert V_fp4x2.dtype == torch.float4_e2m1fn_x2
    assert V_sc_unswizzled.dtype == torch.float8_e4m3fn
    assert V_sc_global.dtype == torch.float32
    assert V_fp4x2.shape[0] == V_sc_unswizzled.shape[0]
    assert V_fp4x2.shape[1] == V_sc_unswizzled.shape[1] * 16 // 2
    assert V_sc_global.shape[0] == 1

    return (
        fp4x2_to_fp32(V_fp4x2) * 
        V_sc_unswizzled.to(torch.float32).repeat_interleave(16, dim=-1) * 
        V_sc_global
    )


def scale_swizzle(
    V_sc_unswizzled: torch.Tensor, # (M, N // 16) fp8e4m3
) -> torch.Tensor:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
    assert len(V_sc_unswizzled.shape) == 2
    assert V_sc_unswizzled.dtype == torch.float8_e4m3fn
    assert V_sc_unswizzled.shape[0] % 128 == 0
    assert (V_sc_unswizzled.shape[1] * 16) % 64 == 0

    M_BLOCK = 128
    N_BLOCK = 4 # 64 / 16

    M, N_16 = V_sc_unswizzled.shape

    V_sc = V_sc_unswizzled        # (M, N_16)
    V_sc = V_sc.reshape(          # (M / 128, 128, N_16 / 4, 4)
        M // M_BLOCK, M_BLOCK,
        N_16 // N_BLOCK, N_BLOCK
    )
    V_sc = V_sc.transpose(1, 2)   # (M / 128, N_16 / 4, 128, 4) --> last 2 dims are all we need per MM
    V_sc = V_sc.reshape(          # (M / 128, N_16 / 4, 4, 32, 4)
        M // M_BLOCK, N_16 // N_BLOCK,
        4, M_BLOCK // 4, N_BLOCK
    )
    V_sc = V_sc.transpose(-2, -3) # (M / 128, N_16 / 4, 32, 4, 4)
    V_sc = V_sc.reshape(          # (M / 128, N_16 / 4, 512)
        M // M_BLOCK, N_16 // N_BLOCK, 512
    )

    return V_sc.contiguous()


def check_diff(
    name: str,
    A: torch.Tensor,
    A_ref: torch.Tensor
) -> None:
    A = fp4x2_to_fp32(A) if A.dtype == torch.float4_e2m1fn_x2 else A.to(torch.float32)
    A_ref = fp4x2_to_fp32(A_ref) if A_ref.dtype == torch.float4_e2m1fn_x2 else A_ref.to(torch.float32)
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
        A_bf16 = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        A_fp4x2_ref, A_sc_unswizzled_ref, A_sc_global_ref = torch_nvfp4_quantize(A_bf16)
        A_sc_ref = scale_swizzle(A_sc_unswizzled_ref)
        A_fp4x2 = torch.empty_like(A_fp4x2_ref)
        A_sc = torch.empty_like(A_sc_ref)
        A_sc_global = torch.empty_like(A_sc_global_ref)
        groups.append((A_bf16, A_fp4x2, A_sc, A_sc_global))

    # Check quantization error (use last group)
    A_bf16_dequantized = torch_nvfp4_dequantize(A_fp4x2_ref, A_sc_unswizzled_ref, A_sc_global_ref)
    check_diff("Quantization error", A_bf16_dequantized, A_bf16)

    # Run our version and check correctness
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global)
    torch.cuda.synchronize()
    check_diff("TK-FP4", A_fp4x2, A_fp4x2_ref)
    check_diff("TK-SC", A_sc, A_sc_ref)
    check_diff("TK-SC-GLOBAL", A_sc_global, A_sc_global_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        nvfp4_quantize(*groups[i % num_groups])
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        nvfp4_quantize(*groups[i % num_groups])
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    gb = M * N * (2 + 2 + 0.5 + 1 / 16) * 1e-9  # bf16 read (absmax + quantize), fp4 & scales write
    gbps = gb / avg_time

    print(f"===============================================================================")
    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average throughput: {gbps:.2f} GB/s")
