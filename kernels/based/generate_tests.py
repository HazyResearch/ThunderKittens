import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 2
H = 2
N = 2048
D = 16
DV = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t0t1', 'ones_t2']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(Q, K, V, add_scale = True, add_norm = False, TESTNAME='all'):
    print("Adding scale:", add_scale)
    B, H, L, D = Q.shape

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))**2
    O2  = make_causal(O)
    T2  = torch.einsum(
        "bhnm,bhmd->bhnd", 
        O2.to(torch.float32), 
        V.to(torch.float32)
    ).to(torch.float32)
    T1a = make_causal(
        torch.einsum("bhnd,bhmd->bhnm", 
        Q.to(torch.float32), K.to(torch.float32))
    )
    T1 = torch.einsum(
        "bhnm,bhme->bhne", 
        T1a.to(torch.float32), V.to(torch.float32)
    ).to(torch.float32)
    T0  = V.to(torch.float32).cumsum(dim=2)

    rd  = math.sqrt(D) if add_scale else 1    
    rrd = math.sqrt(rd) if add_scale else 1   
    r2  = math.sqrt(2) if add_scale else 1

    # KV states
    A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K.to(torch.float32),V.to(torch.float32),K.to(torch.float32)).cumsum(dim=2) / (r2 * rd) 
    A2 = A2[:, :, -1]
    kv_a2 = rearrange(A2, 'b h e f d -> b h (e f) d')
    A1 = torch.einsum("bhnd,bhne->bhnde",K.to(torch.float32),V.to(torch.float32)).cumsum(dim=2)  / rrd
    kv_a1 = A1[:, :, -1].transpose(2, 3)

    o = 0
    o += T0.to(torch.float32)
    o += T1.to(torch.float32) / (rrd * rrd)
    o += T2.to(torch.float32) / (rd * r2 * rd * r2)
    return o.to(torch.bfloat16), kv_a2, kv_a1

o, kv_a2, kv_a1 = pytorch_test(q, k, v, TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().cpu().numpy().tolist()
    kv_a1_cumsumf = kv_a1.to(torch.float32).flatten().cpu().numpy().tolist()
    kv_a2_cumsumf = kv_a2.to(torch.float32).flatten().cpu().numpy().tolist()

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

    # kv states
    for i in trange(B*H*D*DV):
        f.write(repr(kv_a1_cumsumf[i]))
        f.write(' ')
    for i in trange(B*H*D*D*DV):
        f.write(repr(kv_a2_cumsumf[i]))
        f.write(' ')

    