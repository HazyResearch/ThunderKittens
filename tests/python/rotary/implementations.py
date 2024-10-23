import torch
import sys
import os
import time
import argparse
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from rotary.baselines.rotary import RotaryEmbedding
    from rotary.baselines.rotary import apply_rotary_emb
    print(f"Successfully imported RotaryEmbedding")
except:
    RotaryEmbedding = None
    print("Could not import RotaryEmbedding. Please obtain from FlashAttention repo.")
    

#################### Rotary ########################

def get_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*(headdim//2) # mult with cos 
    f += batch*seqlen*nheads*(headdim//2) # mult with sin
    f += batch*seqlen*nheads*(headdim//2) # add rotated values
    return f * 2 # for q and k 


def get_rotary_inputs(b, h, n, dv, dt):
    d_model = h * dv
    head_dim = 128
    rotary_emb_fraction = 1
    rotary_emb_dim = head_dim * rotary_emb_fraction
    rotary_emb_base = 10000
    rotary_max_seqlen = None
    rotary_emb_scale_base=None
    rotary_emb_interleaved=False

    assert RotaryEmbedding is not None, "RotaryEmbedding not imported"

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device='cuda',
    )

    h = d_model // head_dim
    qkv = torch.randn((b, n, 3, h, head_dim), dtype=dt, device='cuda')
    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached
    return qkv, cos, sin, rotary_max_seqlen, rotary_emb


def apply_flash_rotary(dt, b, h, n, dv, verbose=False):
    # Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L957

    qkv, cos, sin, rotary_max_seqlen, rotary_emb  = get_rotary_inputs(b, h, n, dv, dt)
    q = qkv[:,:,0]
    k = qkv[:,:,1]

    torch.cuda.synchronize()
    start = time.time()

    qkv_emb_flash = apply_rotary_emb(
        q, cos, sin
    )

    qkv_emb_flash = apply_rotary_emb(
        k, cos, sin
    )

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start
    return qkv_emb_flash, tot


def apply_rotary_emb_torch(dt, b, h, n, dv, verbose=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """

    qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)

    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    o_dt = qkv.dtype
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= qkv.shape[-1]
    interleaved = False
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]

    torch.cuda.synchronize()
    start = time.time()

    # for q
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    out =  torch.cat(
        [q[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(q[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), q[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)

    # for k 
    out =  torch.cat(
        [k[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(k[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), k[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start 
    return out, tot


def apply_rotary_emb_tk(dt, b, h, n, dv, verbose=False):
    
    qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)
    q = qkv[:,:,0].transpose(1,2).clone()   # torch.Size([b, n, 3, h, dv])
    k = qkv[:,:,1].transpose(1,2).clone()

    cos = cos.contiguous()
    sin = sin.contiguous()
    q = q.contiguous()
    k = k.contiguous()

    torch.cuda.synchronize()
    start = time.time()

    out = tk.fused_rotary(q, cos, sin)
    out = tk.fused_rotary(k, cos, sin)

    torch.cuda.synchronize()
    end = time.time()
    tot = end-start

    assert not np.isnan(out.float().cpu()).any(), "NaN values detected in output 'out'"
    assert not np.isinf(out.float().cpu()).any(), "Inf values detected in output 'out'"
    return out, tot


IMPLEMENTATIONS = {
    'flash_rotary': apply_flash_rotary,
    'rotary_emb_torch': apply_rotary_emb_torch,
    'rotary_emb_tk': apply_rotary_emb_tk
}
