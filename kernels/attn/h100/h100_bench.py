import torch
import numpy as np
import thunderkittens as tk
from flash_attn_interface import flash_attn_func
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def benchmark_attention(configurations, use_fa3=False):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}, {'TK' if not use_fa3 else 'FA3'}")

        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()

        o           = torch.zeros_like(q).contiguous()
        grad_output = torch.randn_like(q, requires_grad=False).contiguous()

        qg = torch.zeros_like(q, requires_grad=False, dtype=torch.float).contiguous()
        kg = torch.zeros_like(k, requires_grad=False, dtype=torch.float).contiguous()
        vg = torch.zeros_like(v, requires_grad=False, dtype=torch.float).contiguous()

        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float, requires_grad=False).contiguous()
        d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float, requires_grad=False).contiguous()
        
        # Prepare for timing forward pass
        start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

        if use_fa3:
            q_ = q.permute(0, 2, 1, 3).contiguous()
            k_ = k.permute(0, 2, 1, 3).contiguous()
            v_ = v.permute(0, 2, 1, 3).contiguous()
            grad_output_ = grad_output.permute(0, 2, 1, 3).contiguous()
            q_.requires_grad = True
            k_.requires_grad = True
            v_.requires_grad = True
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup for forward pass
        for _ in range(10):
            if use_fa3:
                flash_attn_func(q_, k_, v_, causal=causal)
            else:
                tk.mha_forward(q, k, v, causal)
            
        # Time the forward pass
        for i in range(10):          
            start_events_fwd[i].record()
            if use_fa3:
                fa3_out, _ = flash_attn_func(q_, k_, v_, causal=causal)
            else:
                tk.mha_forward(q, k, v, causal)
            end_events_fwd[i].record()

        torch.cuda.synchronize()
        times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        time_us_fwd = np.mean(times_fwd) * 1000

        tflops_fwd = efficiency(flops(B, N, H, D, causal, 'fwd'), time_us_fwd)
        results['fwd'][(D, causal)].append((N, tflops_fwd))

        print(f"Average time for forward pass in us: {time_us_fwd:.2f}")
        print(f"Average efficiency for forward pass in TFLOPS: {tflops_fwd}")
        print("-" * 60)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Prepare for timing backward pass
        start_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        # Warmup for backward pass
        for _ in range(10):
            if use_fa3:
                fa3_out.backward(grad_output_, retain_graph=True)
            else:
                tk.mha_backward(q, k, v, o, l_vec, grad_output, causal)
        
        # Time the backward pass
        for i in range(10):
            start_events_bwd[i].record()
            if use_fa3:
                fa3_out.backward(grad_output_, retain_graph=True)
            else:
                qg, kg, vg = tk.mha_backward(q, k, v, o, l_vec, grad_output, causal)
            end_events_bwd[i].record()

        torch.cuda.synchronize()
        times_bwd = [s.elapsed_time(e) for s, e in zip(start_events_bwd, end_events_bwd)]
        time_us_bwd = np.mean(times_bwd) * 1000

        tflops_bwd = efficiency(flops(B, N, H, D, causal, 'bwd'), time_us_bwd)
        results['bwd'][(D, causal)].append((N, tflops_bwd))

        print(f"Average time for backward pass in us: {time_us_bwd:.2f}")
        print(f"Average efficiency for backward pass in TFLOPS: {tflops_bwd}")
        print("=" * 60)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return results

def plot_results(results_arr, name_arr=['tk']):
    os.makedirs('benchmark_results', exist_ok=True)
    
    # Ensure name_arr has the same length as results_arr
    if len(name_arr) < len(results_arr):
        name_arr.extend([f'method_{i}' for i in range(len(name_arr), len(results_arr))])
    
    for mode in ['fwd', 'bwd']:
        # Collect all configurations across all result sets
        all_configs = set()
        for results in results_arr:
            if mode in results:
                all_configs.update(results[mode].keys())
        
        # Plot each configuration
        for config in all_configs:
            D, causal = config
            
            # Collect data for this configuration from all result sets
            all_seq_lens = set()
            config_data = {}
            
            for i, results in enumerate(results_arr):
                if mode in results and config in results[mode]:
                    values = results[mode][config]
                    seq_lens = [x[0] for x in values]
                    tflops = [x[1] for x in values]
                    
                    # Store data for this method
                    config_data[name_arr[i]] = dict(zip(seq_lens, tflops))
                    all_seq_lens.update(seq_lens)
            
            if not config_data:
                continue
                
            # Sort sequence lengths for consistent ordering
            sorted_seq_lens = sorted(all_seq_lens)
            
            # Prepare data for grouped bar chart
            n_methods = len(config_data)
            n_seq_lens = len(sorted_seq_lens)
            
            # Set up the plot
            plt.figure(figsize=(12, 6))
            
            # Width of each bar and positions
            bar_width = 0.8 / n_methods
            x_positions = np.arange(n_seq_lens)
            
            # Plot bars for each method
            for i, (method_name, method_data) in enumerate(config_data.items()):
                # Get TFLOPS values for this method (0 if seq_len not available)
                tflops_values = [method_data.get(seq_len, 0) for seq_len in sorted_seq_lens]
                
                # Calculate bar positions
                bar_positions = x_positions + i * bar_width - (n_methods - 1) * bar_width / 2
                
                # Create bars
                bars = plt.bar(bar_positions, tflops_values, bar_width, 
                              label=method_name, alpha=0.8)
                
                # Add value labels on bars
                for bar, value in zip(bars, tflops_values):
                    if value > 0:  # Only show label if there's a value
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Customize the plot
            plt.xlabel('Sequence Length')
            plt.ylabel('TFLOPS')
            plt.title(f'{mode.upper()} Pass - Head Dim: {D}, Causal: {causal}')
            plt.xticks(x_positions, sorted_seq_lens)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            filename = f'benchmark_results/{mode}_D{D}_causal{causal}_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

# Example list of configurations to test
configurations = [
    (16, 16, 768,    128, False),
    (16, 16, 768*16,  128, False),
    (16, 16, 768*2,  128, False),
    (16, 16, 768*4,  128, False),
    (16, 16, 768*8,  128, False),
    (16, 16, 768*16, 128, False),
    (16, 16, 768,    128, True),
    (16, 16, 768*2,  128, True),
    (16, 16, 768*4,  128, True),
    (16, 16, 768*8,  128, True),
    (16, 16, 768*16, 128, True),
    (16, 32, 768,    64,  False),
    (16, 32, 768*2,  64,  False),
    (16, 32, 768*4,  64,  False),
    (16, 32, 768*8,  64,  False),
    (16, 32, 768*16, 64,  False),
    (16, 32, 768,    64,  True),
    (16, 32, 768*2,  64,  True),
    (16, 32, 768*4,  64,  True),
    (16, 32, 768*8,  64,  True),
    (16, 32, 768*16, 64,  True),
]

results_tk = benchmark_attention(configurations)
# plot_results(results)

results_fa3 = benchmark_attention(configurations, use_fa3=True)
plot_results([results_tk, results_fa3], name_arr=['tk', 'fa3'])