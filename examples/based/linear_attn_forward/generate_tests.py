import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 1024
D = 16
DV = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t2']:
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D
    v = torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV
elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D
    v = torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(Q, K, V, TESTNAME='all'):

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q, K)**2
    O2  = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2, V)
    T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    T1 = torch.einsum("bhnm,bhme->bhne", T1a, V)  
    T0  = V.cumsum(dim=2)
    
    o = 0
    if 'all' in TESTNAME:
        o  = T0 + T1 + T2/2
    elif 't0' in TESTNAME:
        o = T0
    elif 't1' in TESTNAME:
        o = T1
    elif 't2' in TESTNAME:
        o = T2/2
    return o

o = pytorch_test(q, k, v, TESTNAME)

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


