import torch
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')
import hedgehog as mod

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


sys.path.append("/var/cr05_data/sim_data/code/release/based/train/")
from csrc.causal_dot_prod import causal_dot_product


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
    return out


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
    kv_state = torch.zeros((b, h, d, dv), dtype=dt, device='cuda')
    mod.hedgehog_fwd_tk(Q, K, V, o, kv_state)

    breakpoint()

    return o


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

    pytorch_test_result = pytorch_test(dt, Q, K, V)
    # pytorch_2_test_result = torch_linear_attn(dt, Q, K, V)
    # pytorch_3_test_result = torch_chunk_linear_attn(Q, K, V)
    # fast_transformer_test_result = fast_transformer_test(dt, Q, K, V)
    hedgehog_kernel_test_result = hedgehog_kernel_test(dt, Q, K, V)

    __eq(
        "PyTorch Test v1 - Based Kernel Test", 
        pytorch_test_result[0], 
        hedgehog_kernel_test_result[0], 
        debug=False
    )
    # __eq(
    #     "PyTorch Test v2 - Based Kernel Test", 
    #     pytorch_2_test_result[0], 
    #     hedgehog_kernel_test_result[0], 
    #     debug=False
    # )
    # __eq(
    #     "PyTorch Test v3 - Based Kernel Test", 
    #     pytorch_3_test_result[0], 
    #     hedgehog_kernel_test_result[0], 
    #     debug=False
    # )   
    # __eq(
    #     "Fast Transformer Test - Based Kernel Test", 
    #     fast_transformer_test_result[0], 
    #     hedgehog_kernel_test_result[0], 
    #     debug=False
    # )


print("Correctness test...")
linear_attn_correct(torch.bfloat16)


