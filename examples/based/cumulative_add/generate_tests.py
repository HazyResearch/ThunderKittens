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
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    tensor = torch.arange(N, dtype=torch.float32, device='cuda').repeat_interleave(D).reshape(N, D)
    k = tensor.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
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
    k_a1_cumsum =  K.to(torch.float32).cumsum(dim=2)
    return k_a1_cumsum

k_a1_cumsum = pytorch_test(q, k, v, TESTNAME=TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    kf = k.to(torch.float32).flatten().cpu().numpy()
    k_a1_cumsumf = k_a1_cumsum.to(torch.float32).flatten().cpu().numpy()

    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(k_a1_cumsumf[i]))
        f.write(' ')
