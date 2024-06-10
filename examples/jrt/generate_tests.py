import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 4096
D = 16
DV = 64

# take in N as a command line argument
N = int(sys.argv[1])

# take in dec_enc_ratio as a command line argument
dec_enc_ratio = 2
dec_enc_ratio = int(sys.argv[2]) if len(sys.argv) > 2 else dec_enc_ratio

# assert that N is divisible by dec_enc_ratio
assert N % dec_enc_ratio == 0


torch.random.manual_seed(42)
q     = (torch.randn((B, H, N, D),  dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k     = (torch.randn((B, H, N, D),  dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v     = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)

k_enc = (torch.randn((B, H, N//dec_enc_ratio, D),  dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v_enc = (torch.randn((B, H, N//dec_enc_ratio, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)

    
def pytorch_test(Q, K, V, K_ENC, V_ENC, TESTNAME='all'):
    
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X
    
    # compute decoder attention
    O   = torch.einsum("bhnd,bhmd->bhnm", Q, K)**2
    O2  = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2, V).to(torch.bfloat16).to(torch.float32)
    
    T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    T1 = torch.einsum("bhnm,bhme->bhne", T1a, V).to(torch.bfloat16).to(torch.float32)
    
    T0  = V.cumsum(dim=2).to(torch.bfloat16).to(torch.float32)
    
    o = 0
    if 't0' in TESTNAME or 'all' in TESTNAME:
        o += T0
        print('Adding T0')
    if 't1' in TESTNAME or 'all' in TESTNAME:
        o += T1
        print('Adding T1')
    if 't2' in TESTNAME or 'all' in TESTNAME:
        o += T2/2
        print('Adding T2/2')
        
    # now add in encoder attention
    O_ENC = torch.einsum("bhnd,bhmd->bhnm", Q, K_ENC)**2
    # no need to make causal
    T2_ENC  = torch.einsum("bhnm,bhmd->bhnd", O_ENC, V_ENC).to(torch.bfloat16).to(torch.float32)
    
    T1a_ENC = torch.einsum("bhnd,bhmd->bhnm", Q, K_ENC) # no need to make causal
    T1_ENC = torch.einsum("bhnm,bhme->bhne", T1a_ENC, V_ENC).to(torch.bfloat16).to(torch.float32)
    
    T0_ENC = torch.ones_like(T1a_ENC)
    T0_ENC = torch.einsum("bhnm,bhme->bhne", T0_ENC, V_ENC).to(torch.bfloat16).to(torch.float32)
    
    if 't0' in TESTNAME or 'all' in TESTNAME:
        o += T0_ENC
        print('Adding T0_ENC')
    if 't1' in TESTNAME or 'all' in TESTNAME:
        o += T1_ENC
        print('Adding T1_ENC')
    if 't2' in TESTNAME or 'all' in TESTNAME:
        o += T2_ENC/2
        print('Adding T2_ENC/2')
    
    return o.to(torch.bfloat16)

o = pytorch_test(q, k, v, k_enc, v_enc)

with open(f'randn_all.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    
    k_enc_f = k_enc.to(torch.float32).flatten().cpu().numpy()
    v_enc_f = v_enc.to(torch.float32).flatten().cpu().numpy()
    
    of = o.to(torch.float32).flatten().cpu().numpy()
    
    for i in range(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in range(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in range(B*H*N*DV):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in range(B*H*N*DV):
        f.write(repr(of[i]))
        f.write(' ')
    for i in range(B*H*N*D//dec_enc_ratio):
        f.write(repr(k_enc_f[i]))
        f.write(' ')
    for i in range(B*H*N*DV//dec_enc_ratio):
        f.write(repr(v_enc_f[i]))
        f.write(' ')
    

