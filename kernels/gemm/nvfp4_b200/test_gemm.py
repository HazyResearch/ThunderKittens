import numpy as np
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_gemm, nvfp4_quantize

# TEMP TEMP TEMP TEMP
from test_quantize import torch_nvfp4_quantize, scale_swizzle, torch_nvfp4_dequantize


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
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    print(f"{M=}, {N=}, {K=}")

    # Generate input and output matrices
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = torch.matmul(A, B.T).to(torch.bfloat16)

    # Quantize matrices
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.zeros(M // 128, K // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.zeros(N // 128, K // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
    # nvfp4_quantize(A, A_fp4x2, A_sc, A_sc_global)
    # nvfp4_quantize(B, B_fp4x2, B_sc, B_sc_global)
    # torch.cuda.synchronize()


    # TEMP TEMP TEMP TEMP
    A_fp4x2, A_sc_unswizzled, A_sc_global = torch_nvfp4_quantize(A)
    A_sc = scale_swizzle(A_sc_unswizzled)
    B_fp4x2, B_sc_unswizzled, B_sc_global = torch_nvfp4_quantize(B)
    B_sc = scale_swizzle(B_sc_unswizzled)



    # Run PyTorch version
    A_torch = torch_nvfp4_dequantize(A_fp4x2, A_sc_unswizzled, A_sc_global)
    B_torch = torch_nvfp4_dequantize(B_fp4x2, B_sc_unswizzled, B_sc_global)
    C_torch = torch.matmul(A_torch, B_torch.T)

    # Run kernel and check correctness
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    torch.cuda.synchronize()
    check_diff("TK NVFP4 vs PyTorch NVFP4", C, C_torch)
    check_diff("TK NVFP4 vs PyTorch BF16", C, C_ref)

    quit()

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

    for i in range(NUM_WARMUPS):
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)

    l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
    l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
    cache_clear = lambda: l2_cache.random_(0, 1)

    for i in range(NUM_ITERS):
        cache_clear()
        start_events[i].record()
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12

    print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
    print(f"Average TFLOPs: {tflops / (avg_time):.2f} TFLOp/s")
