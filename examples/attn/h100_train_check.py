import torch
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-3.10')
import h100_train as tk_train

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
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def h100_fwd_kernel_test(Q, K, V, dO):
    o = torch.zeros_like(Q)
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=Q.dtype)
    tk_train.attention_train_forward(Q, K, V, o, l_vec)

    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=Q.dtype)
    qg = torch.zeros_like(Q)
    kg = torch.zeros_like(K)
    vg = torch.zeros_like(V)

    tk_train.attention_train_backward(Q, K, V, o, l_vec, d_vec, dO, qg, kg, vg)

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
    (2, 8, 256),
    (4, 8, 512),
    (8, 8, 1024),
    (16, 8, 2048),
    (16, 8, 4096),
    (16, 8, 8192),
    (1, 1, 16384)
]
for b, h, n in configurations:
    check_correctness(b, h, n)
