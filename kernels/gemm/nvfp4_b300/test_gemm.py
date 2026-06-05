import sys
import torch

torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import nvfp4_gemm, fp32_to_fp4x2, fp4x2_to_fp32  # type: ignore


MMA_PER_TILE = 4
TILE_K = 96 * MMA_PER_TILE


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    print("===============================================================================")
    print(f"<{name}>")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    assert M % 256 == 0
    assert N % 256 == 0
    assert K % TILE_K == 0

    l2_size = 128 * 1024 * 1024
    size_per_group = (M * K + K * N) // 2
    num_groups = (l2_size // size_per_group + 1) * 5
    print(f"{M=}, {N=}, {K=}, {num_groups=}")

    one_e8m0 = torch.full(
        (1,), 0x7F, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)[0]

    groups = []
    for i in range(num_groups):
        A = torch.ones(M, K, dtype=torch.float32, device="cuda")
        B = torch.ones(N, K, dtype=torch.float32, device="cuda")
        A_fp4x2 = fp32_to_fp4x2(A)
        B_fp4x2 = fp32_to_fp4x2(B)
        A_sc = torch.empty(
            M // 128, K // TILE_K, 32, 16 * MMA_PER_TILE,
            dtype=torch.float8_e8m0fnu, device="cuda",
        )
        B_sc = torch.empty(
            N // 128, K // TILE_K, 32, 16 * MMA_PER_TILE,
            dtype=torch.float8_e8m0fnu, device="cuda",
        )
        A_sc.fill_(one_e8m0)
        B_sc.fill_(one_e8m0)
        A_sc_global = torch.ones(1, dtype=torch.float32, device="cuda")
        B_sc_global = torch.ones(1, dtype=torch.float32, device="cuda")
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        groups.append((A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C))

    nvfp4_gemm(*groups[-1])
    A_ref = fp4x2_to_fp32(groups[-1][0])
    B_ref = fp4x2_to_fp32(groups[-1][3])
    check_diff("TK B300 NVFP4 K96 vs PyTorch", groups[-1][-1], torch.matmul(A_ref, B_ref.T).to(torch.bfloat16))

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
    print(f"Average TFLOPs: {tflops / avg_time:.2f} TFLOp/s")
