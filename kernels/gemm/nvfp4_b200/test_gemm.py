import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore


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
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16384

    # Group size
    l2_size = 128 * 1024 * 1024
    size_per_group = (M * K + K * N) // 2
    num_groups = (l2_size // size_per_group + 1) * 5
    print(f"{M=}, {N=}, {K=}, {num_groups=}")

    # Generate input and output matrices and quantize them
    groups = []
    for i in range(num_groups):
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Quantize matrices
        A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
        A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
        B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
        B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
        nvfp4_quantize(A, A_fp4x2, A_sc, A_sc_global, False)
        nvfp4_quantize(B, B_fp4x2, B_sc, B_sc_global, False)

        groups.append((A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C))

    # Run kernel and check correctness using the last input
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    check_diff("TK NVFP4 vs PyTorch BF16", C, torch.matmul(A, B.T).to(torch.bfloat16))

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        nvfp4_gemm(*groups[i % num_groups])
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        nvfp4_gemm(*groups[i % num_groups])
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12

    print(f"Average time: {avg_time * 1e6:.2f} us")
    print(f"Average TFLOPs: {tflops / (avg_time):.2f} TFLOp/s")
