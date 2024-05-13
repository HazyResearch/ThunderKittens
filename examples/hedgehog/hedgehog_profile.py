import torch
import sys
import os
import time

# These installs are for pulling in the TK source
sys.path.append('../../../')
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')

try:
    import hedgehog as mod
    print(f"Succesfully imported hedgehog kernel")
except:
    print(f"Failed to import hedgehog kernel")

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    sys.path.append("../based/linear_attn_forward/")
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Succesfully imported based kernel")
except:
    print(f"Failed to import based kernel")


def pytorch_test(dt, Q, K, V):
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X
    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    o = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)
    return o


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


def torch_linear_attn(dt, q, k, v):
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    return out, last_kv_state


def fast_transformer_test(dt, q, k, v):
    y = causal_dot_product(
        q.contiguous().to(dtype=torch.float32), 
        k.contiguous().to(dtype=torch.float32),
        v.contiguous().to(dtype=torch.float32),
    )
    return y


def hedgehog_kernel_test(dt, Q, K, V,verbose=True):
    q, k, v = Q.unsqueeze(-2), K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state_ref = (k * v).cumsum(dim=2)
    last_kv_state = kv_state_ref[:, :, -1]
    b, h, d, dv = last_kv_state.shape

    o   = torch.zeros_like(V)
    kv_state = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    mod.hedgehog_fwd_tk(Q, K, V, o, kv_state)

    return o, kv_state


def linear_attn_correct(dt):
    b = 2
    n = 1024
    h = 16
    d = 256
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    result_ref_1 = pytorch_test(dt, Q, K, V)
    result_ref_2, kv_state_ref = torch_linear_attn(dt, Q, K, V)
    result_ref_3 = torch_chunk_linear_attn(Q, K, V)
    result_ref_4 = fast_transformer_test(dt, Q, K, V)
    tk_result, tk_kv_state = hedgehog_kernel_test(dt, Q, K, V)

    # Output and KV state
    __eq(
        "PyTorch Test v2 - Based Kernel Test", 
        result_ref_2[0], 
        tk_result[0], 
        debug=False
    )
    __eq(
        "PyTorch Test v2 - Based Kernel Test", 
        kv_state_ref[0], 
        tk_kv_state[0], 
        debug=False
    )

    diff_state = torch.abs(kv_state_ref - tk_kv_state).max()
    print(f"Max diff in state: {diff_state}")

    # Output, more variants
    __eq(
        "PyTorch Test v1 - Based Kernel Test", 
        result_ref_1[0], 
        tk_result[0], 
        debug=False
    )
    __eq(
        "PyTorch Test v3 - Based Kernel Test", 
        result_ref_3[0], 
        tk_result[0], 
        debug=False
    )   
    __eq(
        "Fast Transformer Test - Based Kernel Test", 
        result_ref_4[0], 
        tk_result[0], 
        debug=False
    )


print("Correctness test...")
linear_attn_correct(torch.bfloat16)


