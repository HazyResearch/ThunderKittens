import os
import numpy as np
import math
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
    # To compare to Faster Transformers
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    causal_dot_product = None
    print("Could not import causal_attention_cuda")

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.linear_attn import fused_chunk_linear_attn
    print(f"Successfully imported flash_linear_attention")
except:
    fused_chunk_based, parallel_based = None, None
    print("Could not import flash_linear_attention. pip install -U git+https://github.com/sustcsonglin/flash-linear-attention")


################ Based ################

FEATURE_DIM = 16


def get_flops(batch, seqlen, headdim, nheads):
    expanded_dim = FEATURE_DIM * FEATURE_DIM + FEATURE_DIM + 1
    f = 2 * batch * seqlen * nheads * expanded_dim # compute feature map on q and k
    f += batch * seqlen * nheads * headdim * expanded_dim # (k * v)
    f += batch * seqlen * nheads * headdim * expanded_dim # (cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1)
    return f


def get_based_inputs(b, h, n, dv, dt, seed=42):
    torch.manual_seed(seed)
    Q   = torch.randn((b,h,n,FEATURE_DIM), dtype=dt, device='cuda') / FEATURE_DIM
    K   = torch.randn((b,h,n,FEATURE_DIM), dtype=dt, device='cuda') / FEATURE_DIM
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


def based_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):

    def pytorch_method(q, k, v):
        q, k = feature_map(q), feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2) 
        y = ((q * kv_state).sum(dim=-1)) 
        return y
    
    if torch_compile and method_str == "pytorch":
        pytorch_method = torch.compile(pytorch_method)
    
    for stage in ['warmup', 'timed']:

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            try:
                q, k, v = get_based_inputs(b, h, n, dv, dt)
                feature_map = TaylorExp(input_dim=FEATURE_DIM)

                if method_str == "pytorch":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = pytorch_method(q, k, v)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fast_transformers": 
                    torch.cuda.synchronize()
                    start_events[i].record()
                    q, k = feature_map(q), feature_map(k)
                    y = causal_dot_product(
                        q.contiguous().to(dtype=torch.float32), 
                        k.contiguous().to(dtype=torch.float32),
                        v.contiguous().to(dtype=torch.float32),
                    )
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fla_triton_based":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = parallel_based(q, k, v, False, False)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "tk":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y, kv_state = tk.based( q, k, v )
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert False, f"Unknown method: {method_str}"
            
            except Exception as e:
                return None, -1

            torch.cuda.empty_cache()

    assert not np.isnan(y.float().cpu()).any(), "NaN values detected in output 'y'"
    assert not np.isinf(y.float().cpu()).any(), "Inf values detected in output 'y'"
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    return y, tot


IMPLEMENTATIONS = {
    "pytorch": partial(based_test, causal=True, is_forwards=True, method_str="pytorch"),
    # "fast_transformers": partial(based_test, causal=True, is_forwards=True, method_str="fast_transformers"),
    "fla_triton_based": partial(based_test, causal=True, is_forwards=True, method_str="fla_triton_based"),
    "tk": partial(based_test, causal=True, is_forwards=True, method_str="tk"),
}

NAME = "BASED"
