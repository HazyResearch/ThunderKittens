import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 2048 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 128 if len(sys.argv) <= 3 else int(sys.argv[3])

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    q           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    q = q.to(dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = k.to(dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = v.to(dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    grad_output = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    grad_output = grad_output.to(dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
    v.requires_grad = True
    
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
else:
    print('Invalid test name')
    sys.exit(0)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
o.backward(grad_output)

fn = f'{TESTNAME}_{N}_{D}_bwd.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.detach().to(torch.float32).flatten().cpu().numpy()
    kf = k.detach().to(torch.float32).flatten().cpu().numpy()
    vf = v.detach().to(torch.float32).flatten().cpu().numpy()
    grad_outputf = grad_output.detach().to(torch.float32).flatten().cpu().numpy()
    
    # outputs
    q_grad = q.grad.detach().to(torch.float32).flatten().cpu().numpy()
    k_grad = k.grad.detach().to(torch.float32).flatten().cpu().numpy()
    v_grad = v.grad.detach().to(torch.float32).flatten().cpu().numpy()
    
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
