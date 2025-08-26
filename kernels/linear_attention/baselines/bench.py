import torch
import numpy as np
from lightning_attn2 import lightning_attn2

def benchmark_attention(configurations):    
    for B, H, N, D in configurations:
        print("=" * 60)
        print(f"Timing forward pass for B={B}, H={H}, N={N}, D={D}")

        # Initialize input tensors
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda').contiguous()
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda').contiguous()
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda').contiguous()
        s = torch.rand(H, dtype=torch.float32, device='cuda').contiguous()

        # Prepare timing events
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            _ = lightning_attn2(q, k, v, s)
            
        # Benchmark runs
        print("Running benchmarks...")
        for i in range(10):          
            start_events[i].record()
            _ = lightning_attn2(q, k, v, s)
            end_events[i].record()

        torch.cuda.synchronize()
        
        # Calculate timing statistics
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        time_us = np.mean(times) * 1000  # convert to microseconds
        time_std = np.std(times) * 1000

        print(f"Average latency: {time_us:.2f}us (std: {time_std:.2f}us)")
        print("-" * 60)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    configurations = [
        (1, 8, 1024,  128),
        (1, 8, 2048,  128),
        (1, 8, 4096,  128),
        (1, 8, 8192,  128),
        (1, 8, 16384, 128)
    ]

    print("Linear Attention Benchmark")
    print("=" * 60)
    
    try:
        benchmark_attention(configurations)
        print("\nBenchmark complete!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nOut of memory error. Try reducing batch size or sequence length.")
        else:
            print(f"\nError during benchmark: {str(e)}")