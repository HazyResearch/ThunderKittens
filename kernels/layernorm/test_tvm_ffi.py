from pathlib import Path

import torch
import torch.nn.functional as F
import tvm_ffi


B, N, D = 2, 16, 1024


mod = tvm_ffi.load_module(
    str(Path(__file__).with_name("layernorm_tvm_ffi.so").resolve())
)
torch.manual_seed(0)
x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(D, device="cuda", dtype=torch.bfloat16)
output = torch.empty_like(x)
output_residual = torch.empty_like(x)

mod.tk_layernorm(output, output_residual, x, residual, weight, bias)
torch.cuda.synchronize()

expected_residual = x + residual
expected = F.layer_norm(
    expected_residual.float(), (D,), weight.float(), bias.float(), 1e-5
).to(torch.bfloat16)
torch.testing.assert_close(output_residual, expected_residual, rtol=0.0, atol=0.0)
torch.testing.assert_close(output, expected, rtol=0.05, atol=0.1)
print("TVM-FFI LayerNorm test passed")
