import time
import torch
import matplotlib.pyplot as plt
import os
from einops import rearrange, repeat
from baselines.rotary import RotaryEmbedding

import sys
sys.path.append("kernel/")
import rotary as mod

head_dim = 64
rotary_emb_fraction = 1
rotary_emb_dim = head_dim * rotary_emb_fraction
rotary_emb_base = 10000
rotary_max_seqlen = None
rotary_emb_scale_base=None
rotary_emb_interleaved=False
device='cuda'

n_iters = 100
warmup_iters = 10
dt = torch.bfloat16


def flash_rotary(qkv):
    # Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L957
    torch.cuda.synchronize()
    start = time.time()

    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start
    return tot

def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """

    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    o_dt = x.dtype
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    torch.cuda.synchronize()
    start = time.time()

    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    out =  torch.cat(
        [x[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(x[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), x[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start 
    return tot

def apply_rotary_emb_tk(x, cos_in, sin_in, interleaved=False):
    b, n, h, dh = x.shape 
    out = torch.zeros_like(x)

    torch.cuda.synchronize()
    start = time.time()

    mod.fused_rotary_tk(
        x.contiguous(), 
        cos_in.contiguous(), 
        sin_in.contiguous(), 
        out.contiguous()
    )
    
    torch.cuda.synchronize()
    end = time.time()
    tot = end - start 

    out = rearrange(out, 'b h n d -> b n h d')
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

    if not os.path.exists('plots/'): os.makedirs("plots/")
    plt.savefig(f"plots/{filename}")  # Save the plot
    plt.show()


# batch size performance
timings_flash = []
timings_torch = []
timings_tk = []
batch_sizes = [2, 4, 8, 16, 32]
for b in batch_sizes:

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device=device,
    )

    model_dim = 1024
    h, n, d = int(model_dim/head_dim), 1024, head_dim
    qkv = torch.randn((b, n, 3, h, d), dtype=dt, device=device)
    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached

    torch_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_torch(qkv[:,:,0], cos, sin, interleaved=rotary_emb_interleaved) +  apply_rotary_emb_torch(qkv[:,:,1], cos, sin, interleaved=rotary_emb_interleaved) 
        if i > warmup_iters: torch_iters.append(tot)
    timings_torch.append(sum(torch_iters)/len(torch_iters))

    flash_iters = []
    for i in range(n_iters):
        tot = flash_rotary(qkv) 
        if i > warmup_iters: flash_iters.append(tot)
    timings_flash.append(sum(flash_iters)/len(flash_iters))

    tk_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_tk(qkv[:,:,0].transpose(1,2).clone().detach(), cos, sin) 
        if i > warmup_iters: tk_iters.append(tot)
    timings_tk.append(sum(tk_iters)/len(tk_iters))

plot_timings(
    batch_sizes, timings_torch, timings_flash, timings_tk,
    'Batch Size', 'Time (s)', 'Performance vs Batch Size',
    'performance_vs_batch_size.png'
)

# seqlen performance
timings_flash = []
timings_torch = []
timings_tk = []
seqlens = [1024, 2048, 4096, 8192, 16384, 32768]
for n in seqlens:

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device=device,
    )

    b = 16
    model_dim = 1024
    h, d = int(model_dim/head_dim), head_dim
    qkv = torch.randn((b, n, 3, h, d), dtype=dt, device=device)
    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached

    torch_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_torch(qkv[:,:,0], cos, sin, interleaved=rotary_emb_interleaved) + apply_rotary_emb_torch(qkv[:,:,1], cos, sin, interleaved=rotary_emb_interleaved) 
        if i > warmup_iters: torch_iters.append(tot)
    timings_torch.append(sum(torch_iters)/len(torch_iters))

    flash_iters = []
    for i in range(n_iters):
        tot = flash_rotary(qkv) 
        if i > warmup_iters: flash_iters.append(tot)
    timings_flash.append(sum(flash_iters)/len(flash_iters))

    tk_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_tk(qkv[:,:,0].transpose(1,2).clone().detach(), cos, sin) 
        if i > warmup_iters: tk_iters.append(tot)
    timings_tk.append(sum(tk_iters)/len(tk_iters))

plot_timings(
    seqlens, timings_torch, timings_flash, timings_tk,
    'Seq Len', 'Time (s)', 'Performance vs Seq Len',
                'performance_vs_seq_len.png'
)

# dimension performance
timings_flash = []
timings_torch = []
timings_tk = []
model_dims = [1024, 2048, 4096, 8192]
for model_dim in model_dims:

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device=device,
    )

    b, n = 16, 1024
    h, d = int(model_dim/head_dim), head_dim
    qkv = torch.randn((b, n, 3, h, d), dtype=torch.bfloat16, device=device)
    qkv_emb_flash = rotary_emb(qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen)
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached

    torch_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_torch(qkv[:,:,0], cos, sin, interleaved=rotary_emb_interleaved) + apply_rotary_emb_torch(qkv[:,:,1], cos, sin, interleaved=rotary_emb_interleaved) 
        if i > warmup_iters: torch_iters.append(tot)
    timings_torch.append(sum(torch_iters)/len(torch_iters))

    flash_iters = []
    for i in range(n_iters):
        tot = flash_rotary(qkv) 
        if i > warmup_iters: flash_iters.append(tot)
    timings_flash.append(sum(flash_iters)/len(flash_iters))

    tk_iters = []
    for i in range(n_iters):
        tot = apply_rotary_emb_tk(qkv[:,:,0].transpose(1,2).clone().detach(), cos, sin) 
        if i > warmup_iters: tk_iters.append(tot)
    timings_tk.append(sum(tk_iters)/len(tk_iters))

plot_timings(
    model_dims, timings_torch, timings_flash, timings_tk,
    'Model Dim', 'Time (s)', 'Performance vs Model Dim', 'performance_vs_model_dim.png'
)


