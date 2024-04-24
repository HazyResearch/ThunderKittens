import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 2048
D = 256
DV = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
elif TESTNAME in ['twos']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
elif TESTNAME in ['arange']:
    q = (torch.ones(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV)
    k = (torch.arange(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV*2)
    v = (torch.ones(B*H*N*DV, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, DV)).to(torch.float32)
elif TESTNAME in ['randn']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(Q, K, V, TESTNAME='all'):

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    out = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)

    K, V = K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (K * V).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1]
    return out, last_kv_state

o, last_kv_state = pytorch_test(q, k, v, TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    kv_statef = last_kv_state.to(torch.float32).flatten().cpu().numpy()

    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*DV):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*DV):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*D*DV):
        f.write(repr(kv_statef[i]))
        f.write(' ')


