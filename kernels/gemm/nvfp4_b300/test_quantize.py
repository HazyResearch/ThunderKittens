import sys
import torch

torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import fp32_to_fp4x2, fp4x2_to_fp32  # type: ignore

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


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    assert M % 256 == 0
    assert K % TILE_K == 0

    A = torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.5
    A_fp4x2 = fp32_to_fp4x2(A)
    A_roundtrip = fp4x2_to_fp32(A_fp4x2)

    A_sc = torch.full(
        (M // 128, K // TILE_K, 32, 16 * MMA_PER_TILE),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    check_diff("FP4 roundtrip", A_roundtrip, A)
    print(f"A_fp4x2 shape: {tuple(A_fp4x2.shape)}, dtype: {A_fp4x2.dtype}")
    print(f"A_sc shape: {tuple(A_sc.shape)}, dtype: {A_sc.dtype}")
