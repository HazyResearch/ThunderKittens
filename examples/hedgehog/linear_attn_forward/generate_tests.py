import torch
from tqdm import trange
import numpy as np
import sys

import torch.nn.functional as F

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 4096
D = 128

# accept arg for N 
N = int(sys.argv[1])

torch.random.manual_seed(32)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

# q = q/(float(D*2)**.5)
# k = k/(float(D*2)**.5)
# v = v/(float(D)**.5)

def pytorch_softmax_gt(x):
    
    x_pos = x
    x_neg = -x
    
    x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
    x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
    
    # softmax(x) = torch.exp(x - x_max) / torch.sum(torch.exp(x - x_max), dim=-1) 
    
    x_pos = x_pos - x_pos_max
    x_neg = x_neg + x_neg_max
    
    x_pos_num = torch.exp(x_pos)
    x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
    
    x_neg_num = torch.exp(x_neg)
    x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
    
    x_pos = x_pos_num / x_pos_den
    x_neg = x_neg_num / x_neg_den
    
    x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
    
    return x

def pytorch_test(Q, K, V): 
    
    # q_max = torch.amax(Q, dim=-1, keepdim=True)
    # q_min = torch.amin(Q, dim=-1, keepdim=True)
    # Q = torch.cat([
    #     torch.exp(Q - q_max), torch.exp(-Q + q_min)
    # ], dim=-1)

    # k_max = torch.amax(K, dim=-1, keepdim=True)
    # k_min = torch.amin(K, dim=-1, keepdim=True)
    # K = torch.cat([
    #     torch.exp(K - k_max), torch.exp(-K + k_min)
    # ], dim=-1)
    
    causal = True
    
    a = torch.einsum('bhmd,bhnd->bhmn', Q, K)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    
    # Normalize to compute attention
    # a = a / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(dim=2)))[..., None]
    a = a / (a.sum(dim=-1, keepdim=True))
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    kv_state = torch.einsum('bhlf,bhld->bhfd', K, V).detach()
    
    return out, kv_state, None

# add padding to Q, K, V, and O to make multiple of 64
N = (N + 63) // 64 * 64

q = F.pad(q, (0, 0, 0, N - q.size(2)), value=0)
k = F.pad(k, (0, 0, 0, N - k.size(2)), value=0)
v = F.pad(v, (0, 0, 0, N - v.size(2)), value=0)

q = pytorch_softmax_gt(q)
k = pytorch_softmax_gt(k)
o, kv_state, _ = pytorch_test(q, k, v)

avg_o = torch.mean(torch.abs(o))
avg_kv = torch.mean(torch.abs(kv_state))

print(f"1/100 of Avg mag of o: {avg_o.item()/100}")
print(f"1/100 of Avg mag of kv: {avg_kv.item()/100}")

print("-" * 80)
# print B, H, N, D
print(f'B = {B}')
print(f'H = {H}')
print(f'N = {N}')
print(f'D = {D}')

print("-" * 80)
# print desc of inputs and outputs
print(f'q: {q.shape} {q.dtype}')
print(f'k: {k.shape} {k.dtype}')
print(f'v: {v.shape} {v.dtype}')
print(f'o: {o.shape} {o.dtype}')
print(f'kv_state: {kv_state.shape} {kv_state.dtype}')

print("-" * 80)
with open(f'randn.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    kv = kv_state.to(torch.float32).flatten().cpu().numpy()
    # ksf = k_state.to(torch.float32).flatten().cpu().numpy()
    
    for i in trange(B*H*N*D*2):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D*2):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*2*D*D):
        f.write(repr(kv[i]))
        f.write(' ')
    # for i in trange(B*H*1*2*D):
    #     f.write(repr(ksf[i]))
    #     f.write(' ')

