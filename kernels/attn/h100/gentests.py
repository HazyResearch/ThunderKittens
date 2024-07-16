import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 2048 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 128 if len(sys.argv) <= 3 else int(sys.argv[3])

softmax_scale = 1 / math.sqrt(D)

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    q           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    q = q.to(dtype=torch.bfloat16, device='cuda')
    
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = k.to(dtype=torch.bfloat16, device='cuda')
    
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = v.to(dtype=torch.bfloat16, device='cuda')
    
    grad_output = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    grad_output = grad_output.to(dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
    
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
else:
    print('Invalid test name')
    sys.exit(0)

l_vec = torch.einsum("bhnd,bhmd->bhnm", q.clone(), k.clone()) * softmax_scale
max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec = l_vec.sum(dim=-1, keepdim=True)

l_vec = max_vec + torch.log(l_vec)

q.requires_grad_()
k.requires_grad_()
v.requires_grad_()

o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
o.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

d_vec = torch.mul(grad_output, o)
d_vec = d_vec.sum(dim=-1, keepdim=True)

fn = f'{TESTNAME}_{N}_{D}.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    grad_outputf = grad_output.to(torch.float32).flatten().detach().cpu().numpy()
    
    # intermediate
    l_vecf = l_vec.to(torch.float32).flatten().detach().cpu().numpy()
    
    # outputs
    q_grad = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    k_grad = k_grad.to(torch.float32).flatten().detach().cpu().numpy()
    v_grad = v_grad.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*N):
        f.write(repr(l_vecf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(grad_outputf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(q_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(k_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(v_grad[i]))
        f.write(' ')
        