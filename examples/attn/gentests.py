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
def forward(Q, K, V):
    A0 = torch.einsum("bhnd,bhmd->bhnm",Q, K) * softmax_scale
    
    numerator  = torch.exp(A0) 
    denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
    
    A = numerator / denominator
    
    y  = torch.einsum("bhnm,bhmd->bhnd",A,V)
    return y

def backward(Q, K, V, grad_output):
    running_Q_grad = torch.zeros_like(Q)
    running_K_grad = torch.zeros_like(K)
    running_V_grad = torch.zeros_like(V)
    
    A0 = torch.einsum("bhnd,bhmd->bhnm",Q, K) * softmax_scale
    numerator  = torch.exp(A0) 
    denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
    A = numerator / denominator
        
    running_V_grad += torch.einsum("bhnm,bhnd->bhmd", A, grad_output)
    
    dL_dA = torch.einsum("bhnd,bhmd->bhnm", grad_output, V)
    dL_dA0 = A * (dL_dA - (dL_dA * A).sum(dim=-1, keepdim=True))
    dL_dA0 *= softmax_scale 
    
    running_Q_grad += torch.einsum("bhnm,bhmd->bhnd", dL_dA0, K)
    running_K_grad += torch.einsum("bhnm,bhnd->bhmd", dL_dA0, Q)

    return running_Q_grad, running_K_grad, running_V_grad

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

o = forward(q, k, v)
q_grad, k_grad, v_grad = backward(q, k, v, grad_output)

fn = f'{TESTNAME}_{N}_{D}.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    grad_outputf = grad_output.to(torch.float32).flatten().cpu().numpy()
    
    # outputs
    q_grad = q_grad.to(torch.float32).flatten().cpu().numpy()
    k_grad = k_grad.to(torch.float32).flatten().cpu().numpy()
    v_grad = v_grad.to(torch.float32).flatten().cpu().numpy()
    
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

print(f'Run the harness like `./attn_bwd {fn}`')
