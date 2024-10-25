import sys
import os
import time
import argparse
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    # To compare to Faster Transformers
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    causal_dot_product = None
    print("Could not import causal_attention_cuda")

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    from fla.ops.linear_attn import fused_chunk_linear_attn
    print(f"Successfully imported flash_linear_attention")
except:
    fused_chunk_based, parallel_based, naive_parallel_based = None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention. pip install -U git+https://github.com/sustcsonglin/flash-linear-attention")


################ Based ################


def get_flops(batch, seqlen, headdim, nheads):
    featuredim = 16
    expanded_dim = featuredim * featuredim + featuredim + 1
    f = 2 * batch * seqlen * nheads * expanded_dim # compute feature map on q and k
    f += batch * seqlen * nheads * headdim * expanded_dim # (k * v)
    f += batch * seqlen * nheads * headdim * expanded_dim # (cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1)
    return f


def get_based_inputs(b, h, n, dv, dt, seed=42):
    torch.manual_seed(seed)

    feature_dim = 16
    Q   = torch.randn((b,h,n,feature_dim), dtype=dt, device='cuda') / feature_dim
    K   = torch.randn((b,h,n,feature_dim), dtype=dt, device='cuda') / feature_dim
    V   = torch.randn((b,h,n,dv), dtype=dt, device='cuda') / dv
    return Q, K, V


eps = 1e-12
TERMS = set([0, 1, 2])

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, add_scale=False, **kwargs: any):
        super().__init__()
        self.r2  = math.sqrt(2) if add_scale else 1 
        self.rd  = math.sqrt(input_dim) if add_scale else 1 
        self.rrd = math.sqrt(self.rd) if add_scale else 1 
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor, chosen_terms = [0, 1, 2]):
        terms_list = []
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.to(torch.float32).unsqueeze(-1) * x.to(torch.float32).unsqueeze(-2)).flatten(start_dim=-2) / (self.r2*self.rd)
        terms = [ x[..., :1] ** 0 ]
        terms.append(x / self.rrd)
        terms.append(x2)
        for i, term in enumerate(terms):
            if i in chosen_terms: terms_list.append(term)
        return torch.cat(terms_list, dim=self.head_dim_idx)


def pytorch_test(dt, b, h, n, dv, verbose=True, **kwargs):
    Q, K, v = get_based_inputs(b, h, n, dv, dt)
    d = 16
    feature_map = TaylorExp(input_dim=d)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    
    try:
        torch.cuda.synchronize()
        start_events[0].record()

        q, k = feature_map(Q), feature_map(K)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2)  # causal 
        # k_state = k.cumsum(dim=2)
        y = ((q * kv_state).sum(dim=-1)) 
        
        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    
    except Exception as e:
        tot = -1
        y= None
    return y, tot


def fast_transformer_test(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v = get_based_inputs(b, h, n, dv, dt)
    feature_map = TaylorExp(input_dim=d)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    try:
        torch.cuda.synchronize()
        start_events[0].record()

        q, k = feature_map(q), feature_map(k)
        v = causal_dot_product(
            q.contiguous().to(dtype=torch.float32), 
            k.contiguous().to(dtype=torch.float32),
            v.contiguous().to(dtype=torch.float32),
        )
        y = v 
        
        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]

    except Exception as e:
        tot = -1
        y = None
    return y, tot


def fla_parallel_based_test(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v = get_based_inputs(b, h, n, dv, dt)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    try:
        torch.cuda.synchronize()
        start_events[0].record()

        y = parallel_based(q, k, v, False, False)
        
        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    except Exception as e:
        tot = -1
        y = None
    return y, tot


def based_kernel_test(dt, b, h, n, dv, verbose=True, device='4090'):
    Q, K, V = get_based_inputs(b, h, n, dv, dt)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    try:
        torch.cuda.synchronize()
        start_events[0].record()

        o, kv_state = tk.based( Q, K, V )
        
        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
        assert not np.isnan(o.float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(o.float().cpu()).any(), "Inf values detected in output 'o'"
    except Exception as e:
        tot = -1
        o = None
    return (o, kv_state), tot

    
IMPLEMENTATIONS = {
    "torch_based": pytorch_test,
    # "fast_transformer": fast_transformer_test,
    "fla_triton_based": fla_parallel_based_test,
    "tk_based": based_kernel_test,
}
