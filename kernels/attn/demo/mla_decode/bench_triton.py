import torch
import numpy as np
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from math import sqrt
from torch.nn import init
from torch.nn.functional import linear

from flash_mla import flash_mla_decode, mla_buffers

@dataclass
class BenchConfig:
    batch_size: int
    new_tokens: int
    d_qk: int
    d_vo: int
    page_size: int
    max_length: int
    cache_pages: int
    
    def __str__(self):
        return f"B{self.batch_size}_NT{self.new_tokens}_DQK{self.d_qk}_DVO{self.d_vo}_PS{self.page_size}_ML{self.max_length}_CP{self.cache_pages}"

def mitchell(w: torch.Tensor) -> torch.Tensor:
    """Initialize weights using truncated normal distribution"""
    sigma = 1 / sqrt(w.size(-1))
    bound = 3 * sigma
    init.trunc_normal_(w, mean=0.0, std=sigma, a=-bound, b=bound)
    return w

def run_benchmark(config: BenchConfig, num_warmup: int = 5, num_trials: int = 10):
    """Run benchmark for flash_mla_decode using CUDA events for precise timing"""
    torch.manual_seed(0)
    
    B, R = config.batch_size, config.new_tokens - 1 
    D_QK, D_VO = config.d_qk, config.d_vo
    K, N = config.page_size, config.cache_pages
    MAX_LENGTH = config.max_length
    H = 16  
    
    # Input dimensions
    L = B * (R + 1)
    D = 7_168  # Hidden dimension
    Z = D_QK  # Total head dimension
    Zc = D_VO  # Value dimension
    Zr = Z - Zc  # RoPE dimension
    
    # Sequence length specifics
    T = MAX_LENGTH // K
    Sl, Sh = 0, 32768  # Min/max sequence lengths
    
    # Softmax scale
    Sz = 1 / sqrt(Z)
    
    # init tensors
    hidden = torch.randn(L, D, dtype=torch.bfloat16, device='cuda')
    cache = torch.empty(N, K, 1, Z, dtype=torch.bfloat16, device='cuda')
    weight = mitchell(torch.empty(H * Z, D, dtype=torch.bfloat16, device='cuda'))
    
    query = linear(hidden, weight).view(B, R + 1, H, Z)
    
    # seq lengths and block table
    lengths = torch.randint(Sl, Sh, (B,), dtype=torch.int32, device='cuda')
    sizes = (lengths + (K - 1)).floor_divide(K)
    
    # block table
    table = torch.zeros(B, T, dtype=torch.int32, device='cuda')
    sequence_ids, position_ids = (
        torch.arange(T, dtype=torch.int32, device='cuda')[None, :]
        .expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )
    table[sequence_ids, position_ids] = (
        torch.randperm(N, device='cuda')
        [:sequence_ids.size(0)]
        .sort()
        .values
        .int()
    )
    
    # buffer tensors
    partials, lse, out = mla_buffers(query, T, Zc)
    
    # CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # warmup
    for _ in range(num_warmup):
        flash_mla_decode(
            query=query,
            latent_cache=cache,
            cache_seqlens=lengths,
            block_table=table,
            softmax_scale=Sz,
            latent_dim=Zc,
            rope_dim=Zr,
            partials=partials,
            lse=lse
        )
    
    torch.cuda.synchronize()
    
    # benchmark runs
    for i in range(num_trials):
        start_events[i].record()
        flash_mla_decode(
            query=query,
            latent_cache=cache,
            cache_seqlens=lengths,
            block_table=table,
            softmax_scale=Sz,
            latent_dim=Zc,
            rope_dim=Zr,
            partials=partials,
            lse=lse
        )
        torch.cuda.synchronize()
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    # latencies (converting ms to us)
    latencies_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    avg_latency_us = np.mean(latencies_us)
    
    return avg_latency_us

def plot_results(x_values, y_values, x_label, title, subtitle, output_path):
    plt.figure(figsize=(10, 7))
    
    success_x = []
    success_y = []
    failed_x = []
    
    for x, y in zip(x_values, y_values):
        if y == 0: 
            failed_x.append(x)
        else: 
            success_x.append(x)
            success_y.append(y)
    
    # successful runs with connected lines
    if success_x:
        plt.plot(success_x, success_y, 'o-', linewidth=2, markersize=8, label='Successful runs')
    
    # failed runs with red crosses
    if failed_x:
        # empty list for y-values to place crosses at the bottom
        plt.plot(failed_x, [0] * len(failed_x), 'rx', markersize=10, label='OOM/Failed')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel(x_label)
    plt.ylabel('Latency (μs)')
    
    # y-axis limits with padding for labels
    if success_y:
        plt.ylim(bottom=0, top=max(success_y) * 1.15)
    else:
        plt.ylim(bottom=0, top=100)  # Default range if all runs failed
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.title(subtitle, fontsize=10, pad=10)
    
    # add value labels
    for x, y in zip(success_x, success_y):
        plt.text(x, y + (max(success_y) * 0.02), f'{y:.0f}', ha='center')
    
    # add "OOM" labels for failed runs
    for x in failed_x:
        plt.text(x, 0, 'OOM', ha='center', va='bottom', color='red')
    
    if failed_x:
        plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_configs():
    """Generate different configurations to benchmark"""
    configs = []
    batch_sizes = [1, 4, 8, 16, 18, 32, 64]
    new_tokens = [1, 5, 8, 16, 32]
    cache_pages = [1000, 5000, 10000, 20000, 50000]
    
    base_config = BenchConfig(
        batch_size=18,
        new_tokens=5,
        d_qk=576,
        d_vo=512,
        page_size=256,
        max_length=32768,
        cache_pages=10000
    )
    
    batch_configs = [BenchConfig(
        batch_size=b,
        new_tokens=base_config.new_tokens,
        d_qk=base_config.d_qk,
        d_vo=base_config.d_vo,
        page_size=base_config.page_size,
        max_length=base_config.max_length,
        cache_pages=base_config.cache_pages
    ) for b in batch_sizes]
    
    token_configs = [BenchConfig(
        batch_size=base_config.batch_size,
        new_tokens=t,
        d_qk=base_config.d_qk,
        d_vo=base_config.d_vo,
        page_size=base_config.page_size,
        max_length=base_config.max_length,
        cache_pages=base_config.cache_pages
    ) for t in new_tokens]
    
    cache_configs = [BenchConfig(
        batch_size=base_config.batch_size,
        new_tokens=base_config.new_tokens,
        d_qk=base_config.d_qk,
        d_vo=base_config.d_vo,
        page_size=base_config.page_size,
        max_length=base_config.max_length,
        cache_pages=c
    ) for c in cache_pages]
    
    return {
        'batch': (batch_sizes, batch_configs),
        'tokens': (new_tokens, token_configs),
        'cache': (cache_pages, cache_configs)
    }

def main():
    print("Starting Flash MLA Decode Kernel Benchmark")
    print("-" * 80)
    
    os.makedirs('triton_benchmarks', exist_ok=True)
    
    configs = generate_configs()
    results = {}
    
    for param_name, (x_values, param_configs) in configs.items():
        print(f"\nBenchmarking {param_name} variations")
        latencies = []
        
        for config in param_configs:
            print(f"\nRunning configuration: {config}")
            try:
                latency = run_benchmark(config)
                latencies.append(latency)
                print(f"Latency: {latency:.2f}us")
            except Exception as e:
                print(f"Error running configuration: {e}")
                latencies.append(0)
        
        results[param_name] = (x_values, latencies)
    
    base_config = configs['batch'][1][0]
    
    plot_results(
        results['batch'][0], results['batch'][1],
        'Batch Size', 'Flash MLA Decode Latency vs Batch Size',
        f'Fixed params: NT={base_config.new_tokens}, D_QK={base_config.d_qk}, D_VO={base_config.d_vo}, '
        f'PS={base_config.page_size}, ML={base_config.max_length}, CP={base_config.cache_pages}',
        'triton_benchmarks/batch_size_latency.png'
    )
    
    plot_results(
        results['tokens'][0], results['tokens'][1],
        'New Tokens', 'Flash MLA Decode Latency vs New Tokens',
        f'Fixed params: B={base_config.batch_size}, D_QK={base_config.d_qk}, D_VO={base_config.d_vo}, '
        f'PS={base_config.page_size}, ML={base_config.max_length}, CP={base_config.cache_pages}',
        'triton_benchmarks/new_tokens_latency.png'
    )
    
    plot_results(
        results['cache'][0], results['cache'][1],
        'Cache Pages', 'Flash MLA Decode Latency vs Cache Pages',
        f'Fixed params: B={base_config.batch_size}, NT={base_config.new_tokens}, D_QK={base_config.d_qk}, '
        f'D_VO={base_config.d_vo}, PS={base_config.page_size}, ML={base_config.max_length}',
        'triton_benchmarks/cache_pages_latency.png'
    )

if __name__ == "__main__":
    main()