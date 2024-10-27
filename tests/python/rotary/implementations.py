import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

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


def rotary_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, **kwargs):
    # Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L957

    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
        for i in range(num_iters):
            qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)

            try:
                if method_str == "flash_triton":
                    q = qkv[:,:,0]
                    k = qkv[:,:,1]

                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = apply_rotary_emb(
                        q, cos, sin
                    )
                    y = apply_rotary_emb(
                        k, cos, sin
                    )
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "torch":
                    o_dt = qkv.dtype
                    ro_dim = cos.shape[-1] * 2
                    assert ro_dim <= qkv.shape[-1]
                    interleaved = False
                    q = qkv[:, :, 0]
                    k = qkv[:, :, 1]

                    torch.cuda.synchronize()
                    start_events[i].record()
                    
                    # for q
                    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
                    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
                    y =  torch.cat(
                        [q[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(q[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), q[..., ro_dim:].to(dt)],
                        dim=-1,
                    ).to(o_dt)

                    # for k 
                    y =  torch.cat(
                        [k[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(k[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), k[..., ro_dim:].to(dt)],
                        dim=-1,
                    ).to(o_dt)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "tk":
                    q = qkv[:,:,0].transpose(1,2).clone()   # torch.Size([b, n, 3, h, dv])
                    k = qkv[:,:,1].transpose(1,2).clone()
                    cos = cos.contiguous()
                    sin = sin.contiguous()
                    q = q.contiguous()
                    k = k.contiguous()

                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = tk.fused_rotary(q, cos, sin)
                    y = tk.fused_rotary(k, cos, sin)
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

            except Exception as e:
                if verbose:
                    print(f"Error in rotary_test: {e}")
                return None, -1

            torch.cuda.empty_cache()

    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'out'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'out'"
    return y, tot


IMPLEMENTATIONS = {
    "flash_triton": partial(rotary_test, causal=True, is_forwards=True, method_str="flash_triton"),
    "torch": partial(rotary_test, causal=True, is_forwards=True, method_str="torch"),
    "tk": partial(rotary_test, causal=True, is_forwards=True, method_str="tk"),
}

NAME = "ROTARY"
