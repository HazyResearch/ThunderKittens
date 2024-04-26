

# -*- coding: utf-8 -*-

import torch
from einops import rearrange

import sys
sys.path.append("/var/cr05_data/sim_data/code/release/based/train/")
from csrc.causal_dot_prod import causal_dot_product

def fast_linear_attn(q, k, v):
    y = causal_dot_product(q.contiguous().to(dtype=torch.float32).cuda(), 
                               k.contiguous().to(dtype=torch.float32).cuda(), 
                               v.contiguous().to(dtype=torch.float32).cuda()).to(dtype=q.dtype).cpu()
    return y


def torch_chunk_linear_attn(q, k, v, chunk_size=64):
    q = rearrange(q, 'b h (n c) d -> b h n c d', c = chunk_size) #* (q.shape[-1] **-0.5)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c = chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c = chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([
        torch.zeros_like(kv[:, :, :1]),
        kv[:, :, :-1]
    ], dim=2)
    inter = q @ kv
    intra = ((q @ k.transpose(-1, -2)).masked_fill_(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)) @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')


def torch_linear_attn(q, k, v):

    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)

    return out


def torch_einsum(q, k, v):

    kv_state = torch.einsum("bhnf,bhnd->bhnfd", k, v).cumsum(dim=2)
    out = torch.einsum("bhnf,bhnfd->bhnd", q, kv_state)

    return out

def for_loop(q, k, v):
    B, H, N, D = q.shape
    DV = v.shape[-1]
    o = torch.zeros(B, H, N, DV, dtype=q.dtype, device=q.device)
    b, h = 0, 0
    for n in range(N):
        for d in range(DV):
            for i in range(n):
                o[b, h, n, d] += q[b, h, n, i] * k[b, h, n, i] * v[b, h, n, d]
    return o


if __name__ == '__main__':

    feature_dim = 256
    batch_size = 1
    head_num = 1
    sequence_length = 64
    head_dim = 64
    q = torch.ones(batch_size, head_num, sequence_length, feature_dim)/feature_dim
    k = torch.ones(batch_size, head_num, sequence_length, feature_dim)/feature_dim
    v = torch.ones(batch_size, head_num, sequence_length, head_dim)/head_dim
    
    out_chunk = torch_chunk_linear_attn(q, k, v)
    print(out_chunk.shape)

    out_fast = fast_linear_attn(q, k, v)
    print(out_fast.shape)

    # diff
    diff = torch.norm(out_chunk - out_fast)
    max_diff = torch.max(torch.abs(out_chunk - out_fast))
    print(f"Diff between chunk and fast: {diff}, {max_diff}")

    out_torch = torch_linear_attn(q, k, v)
    print(out_torch.shape)

    # diff
    diff = torch.norm(out_fast - out_torch)
    max_diff = torch.max(torch.abs(out_fast - out_torch))
    print(f"Diff between chunk and fast: {diff}, {max_diff}")

    out_einsum = torch_einsum(q, k, v)
    print(out_einsum.shape)

    # diff
    diff = torch.norm(out_fast - out_einsum)
    max_diff = torch.max(torch.abs(out_fast - out_einsum))
    print(f"Diff between chunk and fast: {diff}, {max_diff}")

    out_loop = for_loop(q, k, v)
    print(out_loop.shape)

    # diff
    diff = torch.norm(out_fast - out_loop)
    max_diff = torch.max(torch.abs(out_fast - out_loop))
    print(f"Diff between chunk and fast: {diff}, {max_diff}")
    
