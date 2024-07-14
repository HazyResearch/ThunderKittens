
import torch 
import torch.nn as nn

import sys
sys.path.append("kernel/")
import layer_norm as mod

def run_tk(x, residual, dropout_p, norm, residual_in_fp32=False):
    x = x.to(dtype=torch.bfloat16)
    residual = residual.to(dtype=torch.bfloat16)
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)

    has_residual = int(residual is not None)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)
    mod.fused_ln_tk(
        int(has_residual), float(dropout_p),
        x, residual, 
        norm_weight, norm_bias, 
        out, out_resid
    )

    return out, out_resid 

b, n, d = 16, 4096, 1024
p = 0.0

torch.manual_seed(0)
x = torch.randn((b, n, d), device='cuda')
residual = torch.randn((b, n, d), device='cuda')

norm = nn.LayerNorm(d).cuda()
fn_out, fn_resid = run_tk(x, residual, p, norm)




