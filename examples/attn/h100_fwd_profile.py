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
    o = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    return o

def h100_fwd_kernel_test(Q, K, V, verbose=True):
    o = torch.zeros_like(Q)
    mod.fwd_attend_ker_tk(Q, K, V, o)
    
    return o

def h100_fwd_correct(): 
    b = 16
    h = 16
    n = 4096
    d = 64
    
    print(f"b={b} h={h} n={n} d={d}")
    
    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda')
    
    result_pytorch = pytorch_test(Q, K, V)
    tk_result = h100_fwd_kernel_test(Q, K, V)
    
    __eq("h100_fwd", result_pytorch[0], tk_result[0], debug=False)
    
print("Correctness test... ")
h100_fwd_correct()