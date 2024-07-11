import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 1
H = 1
N = 1024
D = 64

TESTNAME = sys.argv[1]

norm = nn.LayerNorm(H*D).cuda()

if TESTNAME == 'ones':
    x = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x, norm):
    residual = x
    
    mean = residual.mean(dim=-1, keepdim=True)
    var  = residual.var(dim=-1, keepdim=True, unbiased=False)
    residual_norm = (residual - mean) / torch.sqrt(var + norm.eps)
    o = norm.weight * residual_norm + norm.bias 
    breakpoint()
    return o, mean

o, mean = get_output(x, norm)

with open(f'{TESTNAME}.txt', 'w') as f:
    # inputs
    xf = x.to(torch.float32).flatten().detach().cpu().numpy()
    meanf = mean.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()

    for i in trange(B*H*N*D):
        f.write(repr(xf[i]))
        f.write(' ')

    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    
