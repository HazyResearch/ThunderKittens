import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from torch.nn.functional import scaled_dot_product_attention
import time

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e6   # convert to seconds
    return flop / time

def benchmark_sdpa(configurations, iters=10, dtype=torch.bfloat16):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")
        
        # Create input tensors
        q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
        
        # Create events for timing
        start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup forward pass
        for _ in range(iters):
            with torch.no_grad():
                out = scaled_dot_product_attention(q, k, v, is_causal=causal)
        
        # Time forward pass
        for i in range(iters):
            start_events_fwd[i].record()
            with torch.no_grad():
                out = scaled_dot_product_attention(q, k, v, is_causal=causal)
            end_events_fwd[i].record()
        
        torch.cuda.synchronize()
        times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        time_us_fwd = np.mean(times_fwd) * 1000  # Convert to microseconds
        
        tflops_fwd = efficiency(flops(B, N, H, D, causal, 'fwd'), time_us_fwd)
        results['fwd'][(D, causal)].append((N, tflops_fwd))
        
        print(f"Forward pass - Time: {time_us_fwd:.2f} us, Efficiency: {tflops_fwd:.2f} TFLOPS")
        print("-" * 60)
        
        # # Prepare for backward pass timing
        # start_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        # end_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        # grad_output = torch.randn_like(out)
        
        # # Warmup backward pass
        # for _ in range(iters):
        #     out = scaled_dot_product_attention(q, k, v, is_causal=causal)
        #     out.backward(grad_output, retain_graph=True)
        
        # # Time backward pass
        # for i in range(iters):
        #     # Clear gradients
        #     q.grad = None
        #     k.grad = None
        #     v.grad = None
            
        #     start_events_bwd[i].record()
        #     out = scaled_dot_product_attention(q, k, v, is_causal=causal)
        #     out.backward(grad_output, retain_graph=True)
        #     end_events_bwd[i].record()
        
        # torch.cuda.synchronize()
        # times_bwd = [s.elapsed_time(e) for s, e in zip(start_events_bwd, end_events_bwd)]
        # time_us_bwd = np.mean(times_bwd) * 1000
        
        # tflops_bwd = efficiency(flops(B, N, H, D, causal, 'bwd'), time_us_bwd)
        # results['bwd'][(D, causal)].append((N, tflops_bwd))
        
        # print(f"Backward pass - Time: {time_us_bwd:.2f} us, Efficiency: {tflops_bwd:.2f} TFLOPS")
        # print("=" * 60)
        
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
    
    return results

if __name__ == "__main__":
    # Example configurations
    configurations = [
        # (Batch, Heads, SeqLen, HeadDim, Causal)
        # (16, 16, 768, 128, False),
        # (16, 16, 768*2, 128, False),
        # (16, 16, 768*4, 128, False),
        # (16, 16, 768*8, 128, False),
        (16, 16, 1024, 64, False), 
        (16, 16, 1024, 128, False), 
        # # Add causal versions
        # (16, 16, 768, 128, True),
        # (16, 16, 768*2, 128, True),
        # (16, 16, 768*4, 128, True),
        # (16, 16, 768*8, 128, True),
        # (16, 16, 768*16, 128, True),
    ]
    
    # Run benchmarks
    results = benchmark_sdpa(configurations, iters=10)