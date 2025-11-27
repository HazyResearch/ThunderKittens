import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_gemm, nvfp4_quantize

# TEMPORARY
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
    # Constants. Should equal to kernel's configuration
    PACKED_PER_TILE = 4 

    # Matrix dimensions
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16384

    # Group size
    l2_size = 128 * 1024 * 1024
    size_per_group = M * K + K * N
    num_groups = (l2_size // size_per_group + 1) * 100
    print(f"{M=}, {N=}, {K=}, {PACKED_PER_TILE=}, {num_groups=}")

    # Generate input and output matrices and quantize them
    groups = []
    for i in range(num_groups):
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Quantize matrices
        # A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        # A_sc = torch.zeros(M // 128, K // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda")
        # A_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
        # B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
        # B_sc = torch.zeros(N // 128, K // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda")
        # B_sc_global = torch.zeros(1, dtype=torch.float32, device="cuda")
        # nvfp4_quantize(A, A_fp4x2, A_sc, A_sc_global)
        # nvfp4_quantize(B, B_fp4x2, B_sc, B_sc_global)

        # TEMPORARY
        A_fp4x2, A_sc_unswizzled, A_sc_global = torch_nvfp4_quantize(A)
        A_sc = scale_swizzle(A_sc_unswizzled, PACKED_PER_TILE)
        B_fp4x2, B_sc_unswizzled, B_sc_global = torch_nvfp4_quantize(B)
        B_sc = scale_swizzle(B_sc_unswizzled, PACKED_PER_TILE)

        groups.append((A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C))

    # Run PyTorch version using the last input 
    A_torch = torch_nvfp4_dequantize(A_fp4x2, A_sc_unswizzled, A_sc_global)
    B_torch = torch_nvfp4_dequantize(B_fp4x2, B_sc_unswizzled, B_sc_global)
    C_torch = torch.matmul(A_torch, B_torch.T)

    # Run kernel and check correctness using the last input
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    check_diff("TK NVFP4 vs PyTorch NVFP4", C, C_torch)
    check_diff("TK NVFP4 vs PyTorch BF16", C, torch.matmul(A, B.T).to(torch.bfloat16))

    # Benchmark
    NUM_WARMUPS = 500
    NUM_ITERS = 100

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
