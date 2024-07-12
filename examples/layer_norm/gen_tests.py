import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 1
H = 1
N = 1024
D = 64*H

TESTNAME = sys.argv[1]

norm = nn.LayerNorm(D).cuda()

if TESTNAME == 'ones':
    x = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x, norm):
    residual = x
    
    mean = residual.mean(dim=-1, keepdim=True)
    var  = residual.var(dim=-1, keepdim=True, unbiased=False)
    sqrt_var = torch.sqrt(var + norm.eps)
    residual_norm = (residual - mean) / sqrt_var

    norm_weight = norm.weight
    norm_bias = norm.bias
    norm_weight = torch.randn_like(norm_weight) # was for testing

    o = norm_weight * residual_norm + norm_bias 
    
    return o, norm_weight, norm_bias, mean, sqrt_var

o, norm_weight, norm_bias, mean, var = get_output(x, norm)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy()
    norm_weightf = norm_weight.to(torch.float32).flatten().detach().cpu().numpy()
    norm_biasf = norm_bias.to(torch.float32).flatten().detach().cpu().numpy()
    meanf = mean.to(torch.float32).flatten().detach().cpu().numpy()
    varf = var.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()

    for i in trange(B*N*D):
        f.write(repr(xf[i]))
        f.write(' ')

    for i in trange(B*N*D):
        f.write(repr(of[i]))
        f.write(' ')

    for i in trange(D):
        f.write(repr(norm_weightf[i]))
        f.write(' ')
    
    for i in trange(D):
        f.write(repr(norm_biasf[i]))
        f.write(' ')
    
    for i in trange(B*N*1):
        f.write(repr(meanf[i]))
        f.write(' ')

    for i in trange(B*N*1):
        f.write(repr(varf[i]))
        f.write(' ')

