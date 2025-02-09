import mla_decode
import torch
import numpy as np
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt

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

def run_benchmark(config: BenchConfig, num_warmup: int = 5, num_trials: int = 10):
    """Run benchmark for a specific configuration using CUDA events for precise timing"""
    torch.manual_seed(0)
    
    B, NEW_TOKENS = config.batch_size, config.new_tokens
    D_QK, D_VO = config.d_qk, config.d_vo
    PAGE_SIZE, MAX_LENGTH = config.page_size, config.max_length
    MAX_PAGES = MAX_LENGTH // PAGE_SIZE
    CACHE_PAGES = config.cache_pages

    # Q and O use 16 heads, K and V use 1 head
    Q = torch.randn((B, NEW_TOKENS, 16, D_QK), device='cuda', dtype=torch.bfloat16)
    Cache = torch.randn((CACHE_PAGES, PAGE_SIZE, D_QK), device='cuda', dtype=torch.bfloat16)  # 1 head for K
    Lengths = torch.randint(0, MAX_LENGTH, (B,), device='cuda', dtype=torch.int32)
    Table = torch.randint(0, CACHE_PAGES, (B, MAX_PAGES), device='cuda', dtype=torch.int32)
    O = torch.zeros((B, NEW_TOKENS, 16, D_VO), device='cuda', dtype=torch.bfloat16)  # 16 heads for O
    Sz = 1.0 / (D_QK ** 0.5)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # warmup
    for _ in range(num_warmup):
        mla_decode.mla_decode(Q, Cache, Lengths, Table, O, Sz)
    
    torch.cuda.synchronize()

    # CUDA events
    for i in range(num_trials):
        start_events[i].record()
        mla_decode.mla_decode(Q, Cache, Lengths, Table, O, Sz)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    latencies_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    avg_latency_us = np.mean(latencies_us)
    tokens_per_sec = (B * NEW_TOKENS) / (avg_latency_us / 1e6)
    
    return tokens_per_sec, avg_latency_us

def plot_results(x_values, y_values, x_label, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'o-', linewidth=2, markersize=8)
    plt.grid(True, alpha=0.3)
    plt.xlabel(x_label)
    plt.ylabel('Tokens/sec')
    plt.title(title)
    
    for x, y in zip(x_values, y_values):
        plt.text(x, y + (max(y_values) * 0.02), f'{y:.0f}', ha='center')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_configs():
    """Generate different configurations to benchmark"""
    configs = []
    batch_sizes = [1, 4, 8, 16, 18, 32, 64] 
    new_tokens = [1, 5, 8, 16, 32] 
    cache_pages = [1000, 5000, 10000, 20000, 50000]
    
    # base 
    base_config = BenchConfig(
        batch_size=18,
        new_tokens=5,
        d_qk=576,
        d_vo=512,
        page_size=256,
        max_length=32768,
        cache_pages=10000
    )
    
    # batch size configs
    batch_configs = [BenchConfig(
        batch_size=b,
        new_tokens=base_config.new_tokens,
        d_qk=base_config.d_qk,
        d_vo=base_config.d_vo,
        page_size=base_config.page_size,
        max_length=base_config.max_length,
        cache_pages=base_config.cache_pages
    ) for b in batch_sizes]
    
    # new tokens configs
    token_configs = [BenchConfig(
        batch_size=base_config.batch_size,
        new_tokens=t,
        d_qk=base_config.d_qk,
        d_vo=base_config.d_vo,
        page_size=base_config.page_size,
        max_length=base_config.max_length,
        cache_pages=base_config.cache_pages
    ) for t in new_tokens]
    
    # cache pages configs
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
    print("Starting MLA Decode Kernel Benchmark")
    print("-" * 80)
    
    os.makedirs('tk_benchmarks', exist_ok=True)
    
    configs = generate_configs()
    results = {}
    
    # benchmarks and collect results
    for param_name, (x_values, param_configs) in configs.items():
        print(f"\nBenchmarking {param_name} variations")
        throughputs = []
        
        for config in param_configs:
            print(f"\nRunning configuration: {config}")
            try:
                tokens_per_sec, latency = run_benchmark(config)
                throughputs.append(tokens_per_sec)
                print(f"Tokens/sec: {tokens_per_sec:.2f}")
                print(f"Latency: {latency:.2f}us")
            except Exception as e:
                print(f"Error running configuration: {e}")
                throughputs.append(0)  # Add zero for failed runs to maintain alignment
        
        results[param_name] = (x_values, throughputs)
    
    # plots
    plot_results(
        results['batch'][0], results['batch'][1],
        'Batch Size', 'MLA Decode Performance vs Batch Size',
        'tk_benchmarks/batch_size_perf.png'
    )
    
    plot_results(
        results['tokens'][0], results['tokens'][1],
        'New Tokens', 'MLA Decode Performance vs New Tokens',
        'tk_benchmarks/new_tokens_perf.png'
    )
    
    plot_results(
        results['cache'][0], results['cache'][1],
        'Cache Pages', 'MLA Decode Performance vs Cache Pages',
        'tk_benchmarks/cache_pages_perf.png'
    )

if __name__ == "__main__":
    main()