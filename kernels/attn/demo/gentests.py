import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math

Q_tile = 64
K_tile = 128
def ref_attn(q, k, v, is_causal=False):
    # breakpoint()
    a = q @ k.transpose(-2, -1)
    if is_causal:
        # mask = torch.arange(a.size(-2))[None, None, None, :] > torch.arange(a.size(-1))[None, None, :, None]
        # mask is all True
        mask = torch.full((1, 1, a.size(-2), a.size(-1)), False, device=q.device, dtype=torch.bool)
        for q_idx in range(0, a.size(-2), Q_tile):
            for k_idx in range(0, a.size(-1), K_tile):
                if k_idx > q_idx:
                    mask[:, :, q_idx:q_idx+Q_tile, k_idx:k_idx+K_tile] = True
        # mask = mask.to(q.device)
        # breakpoint()
        a = a.masked_fill(mask, float('-inf'))
    a = torch.nn.functional.softmax(a, dim=-1)
    a = a @ v
    return a

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 2048 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 128 if len(sys.argv) <= 3 else int(sys.argv[3])
causal = False if len(sys.argv) <= 4 else sys.argv[4] == 'causal'
mimic_partial = False if len(sys.argv) <= 5 else sys.argv[5] == 'partial'

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

if mimic_partial:
    o = ref_attn(q, k, v, is_causal=causal)
else:
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

fn = f'{TESTNAME}_{N}_{D}_causal_{causal}_partial_{mimic_partial}.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy() #
    kf = k.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    of = o.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    
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