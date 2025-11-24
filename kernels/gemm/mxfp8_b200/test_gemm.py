import numpy as np
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import mxfp8_gemm, mxfp8_quantize


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
    # Matrix dimensions (should not change)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    K = int(sys.argv[2]) if len(sys.argv) > 3 else 16384
    print(f"{M=}, {N=}, {K=}")

    # Generate input and output matrices
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = torch.matmul(A, B.T).to(torch.bfloat16)

    # Quantize matrices
    A_fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc = torch.zeros(M // 128, K // 128, 512, dtype=torch.uint8, device="cuda")
    B_fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc = torch.zeros(M // 128, K // 128, 512, dtype=torch.uint8, device="cuda")
    mxfp8_quantize(A, A_fp8, A_sc)
    mxfp8_quantize(B, B_fp8, B_sc)
    torch.cuda.synchronize()

    # Run kernel and check correctness
    mxfp8_gemm(A_fp8, A_sc, B_fp8, B_sc, C)
    torch.cuda.synchronize()
    check_diff("TK-MXFP8-GEMM", C, C_ref)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

    for i in range(NUM_WARMUPS):
        mxfp8_gemm(A_fp8, A_sc, B_fp8, B_sc, C)

    l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
    l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
    cache_clear = lambda: l2_cache.random_(0, 1)

    for i in range(NUM_ITERS):
        cache_clear()
        start_events[i].record()
        mxfp8_gemm(A_fp8, A_sc, B_fp8, B_sc, C)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12

    print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
    print(f"Average TFLOPS: {tflops / (avg_time):.2f} TFLOp/s")
