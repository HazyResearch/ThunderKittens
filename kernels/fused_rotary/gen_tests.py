import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange, repeat

B = 1
H = 1
N = 2048
D = 128

D_2 = D // 2

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    x = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
else:
    print('Invalid test name')
    sys.exit(0)


def get_output(x, rotary_emb_base=10000, rotary_emb_dim=D, dtype=torch.bfloat16):

    t = torch.arange(N, device=x.device, dtype=dtype) # We want fp32 here
    inv_freq = 1.0 / (rotary_emb_base ** (torch.arange(0, rotary_emb_dim, 2, device=x.device, dtype=dtype) / rotary_emb_dim))
    freqs = torch.outer(t, inv_freq).to(dtype=dtype)
    cos_in = torch.cos(freqs).to(dtype=dtype)
    sin_in = torch.sin(freqs).to(dtype=dtype)

    ro_dim = cos_in.shape[-1] * 2
    assert ro_dim <= x.shape[-1] 

    ###

    x_ro_dim     = x[..., :ro_dim]
    x_ro_dim_end = x[..., ro_dim:]

    x1, x2 = x_ro_dim.chunk(2, dim=-1)              # D/2, D/2
    rotated_x = torch.cat((-x2, x1), dim=-1)        # D
    
    cos = repeat(cos_in, "n d -> 1 n (2 d)" )
    sin = repeat(sin_in, "n d -> 1 n (2 d)" )
    o = torch.cat([x_ro_dim * cos + rotated_x * sin, x_ro_dim_end], dim=-1)

    # o = x_ro_dim * cos + rotated_x * sin
    # o_ref = torch.concat([-x2*sin_in, x1*sin_in], dim=-1)
    # o_ref = torch.concat([x1*cos_in + -x2*sin_in, x2*cos_in + x1*sin_in], dim=-1)
    # diff = torch.norm(o-o_ref).item()
    # print(f"{diff=}")
    ###

    return o, ro_dim, cos_in, sin_in

o, ro_dim, cos_in, sin_in = get_output(x)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf      = x.to(torch.float32).flatten().detach().cpu().numpy()
    cos_inf = cos_in.to(torch.float32).flatten().detach().cpu().numpy()
    sin_inf = sin_in.to(torch.float32).flatten().detach().cpu().numpy()
    of      = o.to(torch.float32).flatten().detach().cpu().numpy()

    for i in trange(B*H*N*D):
        f.write(repr(xf[i]))
        f.write(' ')

    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')

    for i in trange(N*D_2):
        f.write(repr(cos_inf[i]))
        f.write(' ')

    for i in trange(N*D_2):
        f.write(repr(sin_inf[i]))
        f.write(' ')

print(f"{ro_dim=}")   


    