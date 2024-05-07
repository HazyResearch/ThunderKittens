import torch
import sys
import os
from torch.cuda import Event
from torch.autograd import Function
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-3.10')
import h100_fwd as tk

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AttentionFunction(Function):
    def forward(ctx, q, k, v):
        
        # assert training does not support head dim 128 yet
        assert q.shape[2] == 64, "TK train currently supports head dim 64 only"
        
        outputs = torch.zeros_like(v)
        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=q.dtype).contiguous()
        
        tk.attention_forward(q, k, v, outputs, l_vec)
        
        ctx.save_for_backward(q, k, v, outputs, l_vec)
        
        return outputs

class CustomAttention(nn.Module):
    def __init__(self, b, h, n, d):
        super(CustomAttention, self).__init__()
        self.b = b
        self.h = h
        self.n = n
        self.d = d
        self.scale = 1 / (d ** 0.5)

    def forward(self, q, k, v):
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h).to(dtype=torch.bfloat16).contiguous()
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h).to(dtype=torch.bfloat16).contiguous()
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h).to(dtype=torch.bfloat16).contiguous()
        
        output = AttentionFunction.apply(q, k, v)
        
        output = rearrange(output, 'b h n d -> b n (h d)')
        return output
    
def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def measure_performance(b, h, n, d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "CUDA not available"
    print("Using device:", device)
    att_layer = CustomAttention(b, h, n, d).to(device)

    # Generate random data for q, k, v
    q = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16).contiguous()
    
    q.grad = None
    k.grad = None
    v.grad = None
    
    o = torch.zeros_like(v)
    
    o.grad = None
    
    torch.cuda.synchronize()

    # Warm up
    for _ in range(10):
        tk.attention_forward(q, k, v, o)
        
    
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    for i in range(100):
        # Timing the forward pass
        start_events[i].record()
        
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        tk.attention_forward(q, k, v, o)
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        end_events[i].record()
    
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    time_us = np.mean(times) * 1000
    tflops = efficiency(flops(b, n, d, h, False, "fwd"), time_us)
    
    print(f"Head Dim = {d}, Seq Len = {n}, Heads = {h}, Batch = {b}")
    print(f"Average time taken: {time_us:.2f} us")
    print(f"Efficiency: {tflops:.2f} TFLOPS")
    print(f"______________________________________________________")

# Test configurations
configs = [(32, 16, 1024 * 4, 64), (32, 16, 1024 * 32, 128)]
for config in configs:
    measure_performance(*config)

# Example usage:
# if __name__ == "__main__":
#     # Parameters
#     b, h, n, d = 2, 8, 256, 64  # batch size, heads, sequence length, dimension
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize model and tensors
#     model = CustomAttention(b, h, n, d).to(device)
#     q = torch.randn(b, n, h * d, device=device, dtype=torch.bfloat16)
#     k = torch.randn(b, n, h * d, device=device, dtype=torch.bfloat16)
#     v = torch.randn(b, n, h * d, device=device, dtype=torch.bfloat16)
    
#     # Forward pass
#     output = model(q, k, v)
