import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 4
N = 1024
D = 64*16

TESTNAME = sys.argv[1]

norm = nn.LayerNorm(D).cuda()

if TESTNAME == 'ones':
    x = torch.zeros((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
    residual = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_() * 2
elif TESTNAME == 'arange':
    x = torch.zeros((B, N, D), dtype=torch.bfloat16, device='cuda')
    residual = torch.zeros((B, N, D), dtype=torch.bfloat16, device='cuda')
    for i in range(B):
        for j in range(N):
            x[i,j] = torch.arange(D, dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
    residual = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x, residual, norm, dropout_p=0.00):
    # 1. dropout on x
    mask = torch.bernoulli(torch.full_like(x, 1 - dropout_p))
    dropped = x * mask / (1 - dropout_p) 

    # 3. residual = dropped + residual
    residual = ( dropped + residual ) if residual is not None else dropped 

    # 4. norm
    mean = residual.mean(dim=-1, keepdim=True)
    var  = residual.var(dim=-1, keepdim=True, unbiased=False)
    sqrt_var = torch.sqrt(var + norm.eps)
    residual_norm = (residual - mean) / sqrt_var
    norm_weight = norm.weight
    norm_bias = norm.bias
    # norm_weight = torch.randn_like(norm_weight) # was for testing
    o = norm_weight * residual_norm + norm_bias 
    
    return o, residual, norm_weight, norm_bias, mean, sqrt_var

o, new_residual, norm_weight, norm_bias, mean, var = get_output(x, residual, norm)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    residualf = residual.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    norm_weightf = norm_weight.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    norm_biasf = norm_bias.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    meanf = mean.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    varf = var.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    o_residf = new_residual.to(torch.float32).flatten().detach().cpu().numpy().tolist()

    for i in trange(B*N*D):
        f.write(repr(xf[i]))
        f.write(' ')

    for i in trange(B*N*D):
        f.write(repr(residualf[i]))
        f.write(' ')

    for i in trange(B*N*D):
        f.write(repr(of[i]))
        if (i < 5): print(of[i]) 
        f.write(' ')

    for i in trange(B*N*D):
        f.write(repr(o_residf[i]))
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

