import os
import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    from _C import based
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

class pytorch_based(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = TaylorExp(input_dim=FEATURE_DIM)
        
    def forward(self, q, k, v):
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2) 
        y = ((q * kv_state).sum(dim=-1)) 
        return y
    
def based_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):
    
    pytorch_method = pytorch_based()
    if torch_compile and method_str == "pytorch":
        try:
            pytorch_method = torch.compile(pytorch_method)
        except Exception as e:
            print(f"Could not compile pytorch_method: {e}")
                    
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
                    y, _ = based( q, k, v )
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

################## Common Benchmark Harness ######################

# TODO: this code is redundant on every benchmark currently

import pandas as pd

import warnings
warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

b = 16
h = 16
dv = 64

def efficiency(flops, time):
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms

def measure_efficiency(dt, n, method_name, method, verbose=False, torch_compile=True):
    if verbose:
        print(f"{b=}, {n=}, {h=}, {dv=}")

    if 'c=t' in method_name:
        causal = True
        flops = get_flops(b, n, dv, h, causal=('causal'), mode='bwd' if 'bwd' in method_name else 'fwd')
    elif 'c=f' in method_name:
        causal = False
        flops = get_flops(b, n, dv, h, causal=causal, mode='bwd' if 'bwd' in method_name else 'fwd')
    else:
        flops = get_flops(b, n, dv, h)

    outputs, times = method(dt, b, h, n, dv, verbose=verbose, torch_compile=torch_compile)
    times = times * 1000

    eff = efficiency(flops, times)
    if verbose:
        print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, Time: {times:.4f} us and FLOPS: {flops:.2f}")
    torch.cuda.empty_cache()
    return eff, times

if __name__ == "__main__":
    print("Benchmarking the kernels...")

    verbose = True
    torch_compile = False

    implementations_list = []
    implementations_fwd = IMPLEMENTATIONS
    implementations_list.append(implementations_fwd)
    name = NAME
    print("============" * 4, name, "============" * 4)

    try:
        implementations_bwd = IMPLEMENTATIONS_BWD
        implementations_list.append(implementations_bwd)
    except:
        pass

    for implementations in implementations_list:
        method2tflops = {}
        method2timing = {}

        for m, method in implementations.items():
            flops_result, timing_result = {},  {}
            if verbose:
                print(f"Method: {m}")
            for n in [
                1024 if 'attn' not in m else 768, 
                2048 if 'attn' not in m else 1536, 
                4096 if 'attn' in m else 3072,
                8192 if 'attn' in m else 6144,
                16384 if 'attn' in m else 12288
            ]:
                if "conv" in m and n not in [1024, 4096]:
                    # restrict to sizes we have implemented
                    continue
                if "mamba2_triton" in m and n not in [1024, 2048, 4096, 8192]:
                    # the kernel results in DEVICE_SIDE_ASSERTS
                    continue
                if "layernorm" in m and dv*h != 1024:
                    # restrict to sizes we have implemented
                    print('skipping layernorm due to incompatible model dim')
                if verbose:
                    print(f"Sequence Length: {n}")
                tflops, timing = measure_efficiency(torch.bfloat16, n, m, method, verbose=verbose, torch_compile=torch_compile)
                if tflops > 0: 
                    flops_result[n] = tflops
                    timing_result[n] = timing
            method2tflops[m] = flops_result
            method2timing[m] = timing_result

        # print table
        df = pd.DataFrame(method2tflops).replace(np.nan, 'OOM', regex=True)
        print(df)

