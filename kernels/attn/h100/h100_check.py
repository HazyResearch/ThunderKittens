import torch
import sys
import os
import time


import h100 as tk_train

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pytorch_test(Q, K, V, dO):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def h100_fwd_kernel_test(Q, K, V, dO):
    o = torch.zeros_like(Q)
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
    tk_train.attention_forward_causal(Q, K, V, o, l_vec)

    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
    qg = torch.zeros_like(Q, dtype=torch.float)
    kg = torch.zeros_like(K, dtype=torch.float)
    vg = torch.zeros_like(V, dtype=torch.float)

    tk_train.attention_backward_causal(Q, K, V, o, l_vec, d_vec, dO, qg, kg, vg)

    return o, qg, kg, vg

def check_correctness(b, h, n, d=64):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}")

    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    dO = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()

    pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO)
    tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO)

    results = {
        'Output': (pt_o, tk_o),
        'Q grad': (pt_qg, tk_qg),
        'K grad': (pt_kg, tk_kg),
        'V grad': (pt_vg, tk_vg)
    }

    for key, (pt, tk) in results.items():
        diff = pt - tk
        print(f"{key} - avg magnitude of diff: {torch.mean(torch.abs(diff)).item():.6f}")
        print("-" * 40)

print("Correctness Tests: ")
configurations = [
    (2, 12, 384*2),
    (4, 12, 384*4),
    (8, 12, 384*8),
    (16, 12, 384*16),
    (16, 12, 384*32),
    (16, 12, 384*64),
    (1, 12, 384*128)
]
for b, h, n in configurations:
    check_correctness(b, h, n)