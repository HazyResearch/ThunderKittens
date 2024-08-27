import torch 
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-3.10')
import h100_fwd as mod

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pytorch_test(Q, K, V):
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    return output

def h100_fwd_kernel_test(Q, K, V):
    o = torch.zeros_like(Q)
    mod.attention_forward_causal(Q, K, V, o)
    return o

def check_correctness(b, h, n, d):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}")
    
    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
    
    result_pytorch = pytorch_test(Q, K, V)
    tk_result = h100_fwd_kernel_test(Q, K, V)
    
    diff = result_pytorch - tk_result
    avg_diff_mag = torch.mean(torch.abs(diff)).item()
    avg_diff_per = 100 * avg_diff_mag / torch.mean(torch.abs(result_pytorch)).item()
    
    print(f"Attention output - avg magnitude of diff: {avg_diff_mag:.6f}")
    print("-" * 40)

print("Correctness Tests: ")
configurations = [
    (2,  8, 256,   64),
    (4,  8, 512,   64),
    (8,  8, 1024,  64),
    (16, 8, 2048,  64),
    (16, 8, 4096,  64),
    (16, 8, 8192,  64),
    (16, 8, 16384, 64),
    (2,  8, 256,   128),
    (4,  8, 512,   128),
    (8,  8, 1024,  128),
    (16, 8, 2048,  128),
    (16, 8, 4096,  128),
    (16, 8, 8192,  128),
    (16, 8, 16384, 128)
]
for b, h, n, d in configurations:
    check_correctness(b, h, n, d)
