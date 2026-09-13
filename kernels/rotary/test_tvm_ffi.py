from pathlib import Path

import torch
import tvm_ffi


B, H, N = 2, 3, 144


mod = tvm_ffi.load_module(
    str(Path(__file__).with_name("rotary_tvm_ffi.so").resolve())
)
torch.manual_seed(0)

for D in (64, 128):
    x = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(N, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D)
    )
    frequencies = torch.outer(positions, inv_freq)
    cos = frequencies.cos().to(torch.bfloat16)
    sin = frequencies.sin().to(torch.bfloat16)
    output = torch.empty_like(x)

    mod.tk_rotary(output, x, cos, sin)
    torch.cuda.synchronize()

    x1, x2 = x.float().chunk(2, dim=-1)
    expected = torch.cat(
        (
            x1 * cos.float()[None, None] - x2 * sin.float()[None, None],
            x2 * cos.float()[None, None] + x1 * sin.float()[None, None],
        ),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)

print("TVM-FFI rotary test passed")
