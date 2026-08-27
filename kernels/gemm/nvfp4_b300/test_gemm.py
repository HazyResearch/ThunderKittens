import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_gemm, fp32_to_fp4x2, fp4x2_to_fp32  # type: ignore

# Each K-tile is 8 K96 MMAs wide
MMA_PER_TILE = 8
TILE_K = 96 * MMA_PER_TILE
SF_TILE_ROWS = 24


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
    # Matrix dimensions (K must be a multiple of TILE_K)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16128
    scale_kind = sys.argv[4] if len(sys.argv) > 4 else "e8m0"
    assert M % 256 == 0
    assert N % 256 == 0
    assert K % TILE_K == 0
    assert scale_kind in ("e4m3", "e8m0")

    scale_dtype = torch.float8_e4m3fn if scale_kind == "e4m3" else torch.float8_e8m0fnu
    scale_cols = (32 if scale_kind == "e4m3" else 16) * MMA_PER_TILE

    # Group size
    l2_size = 128 * 1024 * 1024
    size_per_group = (M * K + K * N) // 2
    num_groups = (l2_size // size_per_group + 1) * 5
    print(f"{M=}, {N=}, {K=}, {num_groups=}")

    # Generate input and output matrices and quantize them
    # Unit local scales isolate the packed-FP4 GEMM while non-unit global scales
    # verify that the public extension applies the full NVFP4 scaling contract.
    one_scale = (
        torch.full((1,), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)[0]
        if scale_kind == "e8m0"
        else torch.ones(1, dtype=torch.float8_e4m3fn, device="cuda")[0]
    )
    groups = []
    for i in range(num_groups):
        A = torch.randn(M, K, dtype=torch.float32, device="cuda") / K ** 0.25
        B = torch.randn(N, K, dtype=torch.float32, device="cuda") / K ** 0.25
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Quantize matrices
        A_fp4x2 = fp32_to_fp4x2(A)
        B_fp4x2 = fp32_to_fp4x2(B)
        A_sc = torch.empty(M // 128, K // TILE_K, SF_TILE_ROWS, scale_cols, dtype=scale_dtype, device="cuda")
        B_sc = torch.empty(N // 128, K // TILE_K, SF_TILE_ROWS, scale_cols, dtype=scale_dtype, device="cuda")
        A_sc.fill_(one_scale)
        B_sc.fill_(one_scale)
        A_sc_global = torch.full((1,), 0.5, dtype=torch.float32, device="cuda")
        B_sc_global = torch.full((1,), 0.75, dtype=torch.float32, device="cuda")

        groups.append((A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C))

    # Run kernel and check correctness using the last input
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    A_ref = fp4x2_to_fp32(A_fp4x2)
    B_ref = fp4x2_to_fp32(B_fp4x2)
    scale = A_sc_global * B_sc_global
    check_diff("TK NVFP4 K96 vs PyTorch BF16", C, (torch.matmul(A_ref, B_ref.T) * scale).to(torch.bfloat16))

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
