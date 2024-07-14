import torch 
import time 
import torch.nn as nn
# from torchvision.ops import StochasticDepth
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("kernel/")
import layer_norm as mod

warmup_iters = 10
num_iters = 100


def run_torch(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    timings = []

    for i in range(num_iters):
        torch.cuda.synchronize()
        st = time.time()

        # dropped = drop_path(dropout(x))
        dropped = dropout(x)
        residual = (residual + dropped) if residual is not None else dropped
        x = norm(residual.to(dtype=norm.weight.dtype))
        residual = residual.to(torch.float32)

        torch.cuda.synchronize()
        et = time.time()
        tot = (et - st)
        if i > warmup_iters:
            timings.append(tot)

    tot = sum(timings)/len(timings)
    return tot


def run_flash(x, residual, drop_path, dropout, norm, residual_in_fp32=True):
    from layer_norm_triton import layer_norm_fn, RMSNorm

    timings = []

    for i in range(num_iters):

        torch.cuda.synchronize()
        st = time.time()
        rowscale = torch.ones(
                x.shape[:-1],
                device=x.device,
                dtype=x.dtype,
            )
        # drop_path(
            
        # )
        x, residual = layer_norm_fn(
            x,
            norm.weight,
            norm.bias,
            residual=residual,
            eps=norm.eps,
            dropout_p=dropout.p,
            rowscale=rowscale,
            prenorm=True,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False
        )
        torch.cuda.synchronize()
        et = time.time()
        tot = (et - st)
        if i > warmup_iters:
            timings.append(tot)

    tot = sum(timings)/len(timings)
    return tot


def run_tk(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    x = x.to(dtype=torch.bfloat16)
    residual = residual.to(dtype=torch.bfloat16)
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)

    has_residual = int(residual is not None)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)

    timings = []

    for i in range(num_iters):

        torch.cuda.synchronize()
        st = time.time()
        
        mod.fused_ln_tk(
            int(has_residual), float(dropout.p),
            x, residual, 
            norm_weight, norm_bias, 
            out, out_resid
        )

        torch.cuda.synchronize()
        et = time.time()
        tot = (et - st)
        if i > warmup_iters:
            timings.append(tot)

    tot = sum(timings)/len(timings)
    return tot


def plot_timings(x_values, y_torch, y_flash, y_tk, x_label, y_label, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_torch, marker='o', label='Torch')
    plt.plot(x_values, y_flash, marker='s', label='Flash')
    plt.plot(x_values, y_tk, marker='o', label='TK')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

    plt.savefig(f"plots/{filename}")  # Save the plot
    plt.show()


if __name__ == "__main__":
    p = 0.1

    if not os.path.exists("plots"):
        os.makedirs("plots/")

    # Varying batch size
    n, d = 2048, 1024
    batch_sizes = [1, 2, 4, 8, 16, 32]
    torch_timings_batch = []
    flash_timings_batch = []
    tk_timings_batch = []
    
    for b in batch_sizes:
        torch.manual_seed(0)
        x = torch.randn((b, n, d), device='cuda')
        residual = torch.randn((b, n, d), device='cuda')
        norm = nn.LayerNorm(d).cuda()
        dropout = nn.Dropout(p)
        drop_path = None #StochasticDepth(p, mode="row")
        torch_timings_batch.append(run_torch(x, residual, drop_path, dropout, norm))
        flash_timings_batch.append(run_flash(x, residual, drop_path, dropout, norm))
        tk_timings_batch.append(run_tk(x, residual, drop_path, dropout, norm))
    
    plot_timings(batch_sizes, torch_timings_batch, flash_timings_batch, tk_timings_batch, 
                 'Batch Size', 'Time (s)', 'Performance vs Batch Size', 'batch_size')

    # Varying sequence length
    b, d = 16, 1024
    seq_lengths = [512, 1024, 2048, 4096]
    torch_timings_seq = []
    flash_timings_seq = []
    tk_timings_seq = []
    
    for n in seq_lengths:
        torch.manual_seed(0)
        x = torch.randn((b, n, d), device='cuda')
        residual = torch.randn((b, n, d), device='cuda')
        norm = nn.LayerNorm(d).cuda()
        dropout = nn.Dropout(p)
        drop_path = None #StochasticDepth(p, mode="row")
        torch_timings_seq.append(run_torch(x, residual, drop_path, dropout, norm))
        flash_timings_seq.append(run_flash(x, residual, drop_path, dropout, norm))
        tk_timings_seq.append(run_tk(x, residual, drop_path, dropout, norm))
    
    plot_timings(seq_lengths, torch_timings_seq, flash_timings_seq, tk_timings_seq,
                 'Sequence Length', 'Time (s)', 'Performance vs Sequence Length',
                 'performance_vs_sequence_length.png')

    # Varying dimension
    # b, n = 16, 2048
    # dimensions = [512, 1024, 2048, 4096]
    # torch_timings_dim = []
    # flash_timings_dim = []
    # tk_timings_dim = []
    
    # for d in dimensions:
    #     torch.manual_seed(0)
    #     x = torch.randn((b, n, d), device='cuda')
    #     residual = torch.randn((b, n, d), device='cuda')
    #     norm = nn.LayerNorm(d).cuda()
    #     dropout = nn.Dropout(p)
    #     drop_path = StochasticDepth(p, mode="row")
    #     torch_timings_dim.append(run_torch(x, residual, drop_path, dropout, norm))
    #     flash_timings_dim.append(run_flash(x, residual, drop_path, dropout, norm))
    #     tk_timings_dim.append(run_tk(x, residual, drop_path, dropout, norm))
    
    # plot_timings(dimensions, torch_timings_dim, flash_timings_dim, tk_timings_dim,
    #              'Dimension', 'Time (s)', 'Performance vs Dimension',
    #              'performance_vs_dimension.png')
    

    



