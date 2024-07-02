import torch
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange 

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
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

def pytorch_test(Q, K, V, add_scale = False, add_norm = True, TESTNAME='all'):

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
    ).to(torch.bfloat16).to(torch.float32)
    T1a = make_causal(
        torch.einsum("bhnd,bhmd->bhnm", 
        Q.to(torch.float32), K.to(torch.float32))
    )
    T1 = torch.einsum(
        "bhnm,bhme->bhne", 
        T1a.to(torch.float32), V.to(torch.float32)
    ).to(torch.bfloat16).to(torch.float32)
    T0  = V.cumsum(dim=2).to(torch.bfloat16).to(torch.float32)

    rd = math.sqrt(D) if add_scale else 1 
    rrd = math.sqrt(rd) if add_scale else 1
    r2 = math.sqrt(2) if add_scale else 1

    # Denominator
    K0 = torch.ones(Q[..., :1].to(torch.float32).shape).to(Q.device)
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q.to(torch.float32), Q.to(torch.float32)) / (rd * r2)
    K2 = torch.einsum("bhnd,bhne->bhnde", K.to(torch.float32), K.to(torch.float32)) / (rd * r2)
    k_a2_cumsum = K2.to(torch.float32).cumsum(dim=2)
    D2 = torch.einsum("bhnde,bhnde->bhn", Q2.to(torch.float32), k_a2_cumsum) 
    k_a1_cumsum =  K.to(torch.float32).cumsum(dim=2)
    D1 = torch.einsum("bhnd,bhnd->bhn", Q.to(torch.float32), k_a1_cumsum)/ ((rrd) ** 2)
    k_a0_cumsum =  K0.to(torch.float32).cumsum(dim=2).squeeze(-1)

    o = 0
    den = 0
    if add_norm: 
        norm_a0 = k_a0_cumsum.to(torch.bfloat16).to(torch.float32)
        den += norm_a0
    o += T0.to(torch.bfloat16).to(torch.float32)

    if add_norm: 
        norm_a1 = D1.to(torch.bfloat16).to(torch.float32)
        den += norm_a1
    o += T1.to(torch.bfloat16).to(torch.float32) / (rrd * rrd)
    
    if add_norm: 
        norm_a2 = D2.to(torch.bfloat16).to(torch.float32)
        # den += norm_a2
        pass
    o += T2.to(torch.bfloat16).to(torch.float32) / (rd * r2 * rd * r2)

    if add_norm and not (type(den) == int and den > 0):
        print("normalizing!")
        eps = 1e-12
        o = o / (den.unsqueeze(-1) + eps)

    k_a2_cumsum = rearrange(k_a2_cumsum, 'b h n d e -> b h n (d e)')
    return o.to(torch.bfloat16), k_a2_cumsum[:,:,-1], k_a1_cumsum[:,:,-1], k_a0_cumsum[:,:,-1], norm_a1, norm_a2, k_a1_cumsum, k_a2_cumsum, k_a0_cumsum

o, k_a2, k_a1, k_a0, norm_a1, norm_a2, k_a1_cumsum, k_a2_cumsum, k_a0_cumsum = pytorch_test(q, k, v, TESTNAME=TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    k_a2f = k_a2.to(torch.float32).flatten().cpu().numpy()
    k_a1f = k_a1.to(torch.float32).flatten().cpu().numpy()
    k_a0f = k_a0.to(torch.float32).flatten().cpu().numpy()
    norm_a1f = norm_a1.to(torch.float32).flatten().cpu().numpy()
    norm_a2f = norm_a2.to(torch.float32).flatten().cpu().numpy()
    k_a1_cumsumf = k_a1_cumsum.to(torch.float32).flatten().cpu().numpy()
    k_a2_cumsumf = k_a2_cumsum.to(torch.float32).flatten().cpu().numpy()
    k_a0_cumsumf = k_a0_cumsum.to(torch.float32).flatten().cpu().numpy()
    
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
    for i in trange(B*H*D):
        f.write(repr(k_a1f[i]))
        f.write(' ')
    for i in trange(B*H*D*D):
        f.write(repr(k_a2f[i]))
        f.write(' ')

    for i in trange(B*H*N*D):
        f.write(repr(k_a1_cumsumf[i]))
        f.write(' ')
    for i in trange(B*H*N):
        f.write(repr(norm_a1f[i]))
        f.write(' ')

    for i in trange(B*H*N*D*D):
        f.write(repr(k_a2_cumsumf[i]))
        f.write(' ')
    for i in trange(B*H*N):
        f.write(repr(norm_a2f[i]))
        f.write(' ')

    for i in trange(B*H*N):
        f.write(repr(k_a0_cumsumf[i]))
        f.write(' ')

