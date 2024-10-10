import torch
import h100 as tk
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pytorch_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def h100_fwd_kernel_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    o = torch.zeros_like(Q)
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    
    tk.attention_forward(Q, K, V, o, l_vec, causal)

    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    qg = torch.zeros_like(Q, dtype=torch.float)
    kg = torch.zeros_like(K, dtype=torch.float)
    vg = torch.zeros_like(V, dtype=torch.float)

    tk.attention_backward(Q, K, V, o, l_vec, d_vec, dO, qg, kg, vg, causal)

    return o, qg, kg, vg

def check_correctness(b, h, n, d, causal):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}, causal={causal}")

    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    dO = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()

    pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, causal)
    tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, causal)

    results = {
        'Output': (pt_o, tk_o),
        'Q grad': (pt_qg, tk_qg),
        'K grad': (pt_kg, tk_kg),
        'V grad': (pt_vg, tk_vg)
    }

    for key, (pt, tk) in results.items():
        diff = pt - tk

        tol = torch.mean(torch.abs(pt)).item() * 0.01
        print(f"{key} - max acceptable magnitude of diff: {tol:.6f}")
        print(f"{key} - avg magnitude of diff: {torch.mean(torch.abs(diff)).item():.6f}")
        
        # if avg mag < 1% of avg mag of pt, then pass
        assert(torch.mean(torch.abs(diff)).item() < tol)
        print(f"{key} - pass: {torch.mean(torch.abs(diff)).item() < tol}")
        print("-" * 40)

print("Correctness Tests: ")
configurations = [
    (32, 16, 768,    128, False),
    (16, 16, 768*2,  128, False),
    (8,  16, 768*4,  128, False),
    (4,  16, 768*8,  128, False),
    (2,  16, 768*16, 128, False),
    (1,  16, 768*32, 128, False),
    (32, 16, 768,    128, True),
    (16, 16, 768*2,  128, True),
    (8,  16, 768*4,  128, True),
    (4,  16, 768*8,  128, True),
    (2,  16, 768*16, 128, True),
    (1,  16, 768*32, 128, True), 
    (32, 32, 768,    64,  False),
    (16, 32, 768*2,  64,  False),
    (8,  32, 768*4,  64,  False),
    (4,  32, 768*8,  64,  False),
    (2,  32, 768*16, 64,  False),
    (1,  32, 768*32, 64,  False),
    (32, 32, 768,    64,  True),
    (16, 32, 768*2,  64,  True),
    (8,  32, 768*4,  64,  True),
    (4,  32, 768*8,  64,  True),
    (2,  32, 768*16, 64,  True),
    (1,  32, 768*32, 64,  True),
]

for b, h, n, d, causal in configurations:
    check_correctness(b, h, n, d, causal)
