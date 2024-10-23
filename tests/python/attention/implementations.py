import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
import thunderkittens as tk

try:
    # To compare to Flash Attention
    from flash_attn import flash_attn_func
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention. Please: pip install flash-attn")


################ Attention ################

def get_flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def get_attention_inputs(b, h, n, dv, dt=torch.bfloat16):
    q = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    k = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    v = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    return q, k, v


def flash_attention_2_test(dt, b, h, n, dv, causal, verbose=True, **kwargs):
    q, k, v = get_attention_inputs(b, h, n, dv, dt)
    q = torch.randn_like(v).transpose(1,2)
    k = torch.randn_like(v).transpose(1,2)
    v = torch.randn_like(v).transpose(1,2)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        y = flash_attn_func(
                q, k, v,
                softmax_scale=0.5,
                causal=causal 
            )
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


def benchmark_attention_tk(dt, b, h, n, dv, causal, verbose=True, **kwargs):
    q, k, v = get_attention_inputs(b, h, n, dv, dt)
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        
        y, l_vec = tk.mha_forward(q, k, v, causal)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
        assert not np.isnan(y.float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(y.float().cpu()).any(), "Inf values detected in output 'o'"
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot
    

from functools import partial
IMPLEMENTATIONS = {
    "fwd_fa2_causalTrue": partial(flash_attention_2_test, causal=True),
    "fwd_tk_causalTrue": partial(benchmark_attention_tk, causal=True),
    "fwd_fa2_causalFalse": partial(flash_attention_2_test, causal=False),
    "fwd_tk_causalFalse": partial(benchmark_attention_tk, causal=False),
}


