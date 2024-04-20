import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 4096
D = 64

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
else:
    print('Invalid test name')
    sys.exit(0)

att = torch.einsum('bhtd,bhsd->bhts', q.to(torch.float32), k.to(torch.float32))
att /= D**.5
att = torch.exp(att)
for t in range(N):
    att[:,:,t,t+1:] = 0 # make causal
    if t > 64:
        att[:,:,t,:16*((t-64)//16)] = 0 # make sliding
att /= att.sum(dim=-1, keepdims=True) # normalize softmax
o = torch.einsum('bhts,bhsd->bhtd', att.to(torch.bfloat16), v)
# o = torch.nn.functional.scaled_dot_product_attention(q, k, v)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
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

print(f'Run the harness like `cat {TESTNAME}.txt | ./harness`')