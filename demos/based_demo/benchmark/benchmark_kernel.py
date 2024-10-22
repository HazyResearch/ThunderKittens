import torch
import sys
import os
import time
import argparse

import thunderkittens as tk
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn

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
    print(f"Successfully imported flash_linear_attention")
except:
    fused_chunk_based, parallel_based, naive_parallel_based = None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention. Install from https://github.com/sustcsonglin/flash-linear-attention")

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except:
    mamba_chunk_scan_combined = None 
    print("Could not import mamba_chunk_scan_combined. Please: pip install mamba_ssm")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print(f"Successfully imported selective_scan_fn")
except:
    selective_scan_fn = None 
    print("Could not import selective_scan_fn. Please: pip install mamba_ssm")

try:
    # To compare to Flash Attention
    from flash_attn import flash_attn_func
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention. Please: pip install flash-attn")


################ Flash Attention ################

def flash_attention_test(dt, q, k, v, d, verbose=True, **kwargs):

    try:
        q = torch.randn_like(v).transpose(1,2)
        k = torch.randn_like(v).transpose(1,2)
        v = torch.randn_like(v).transpose(1,2)

        torch.cuda.synchronize()
        t0 = time.time()

        y = flash_attn_func(
            q, k, v,
            softmax_scale=0.5,
            causal=True, 
        )
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, **kwargs: any):
        super().__init__()
        import math
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)
        self.rrd = math.sqrt(self.rd)
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)


def pytorch_test(dt, Q, K, v, d, verbose=True, **kwargs):
    feature_map = TaylorExp(input_dim=d)
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        q, k = feature_map(Q), feature_map(K)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2)  # causal 
        k_state = k.cumsum(dim=2)
        y = ((q * kv_state).sum(dim=-1)) 

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}") # likely OOM
        tot = -1
        y= None
    return y, tot


def fast_transformer_test(dt, q, k, v, d, verbose=True, **kwargs):
    feature_map = TaylorExp(input_dim=d)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        q, k = feature_map(q), feature_map(k)
        v = causal_dot_product(
            q.contiguous().to(dtype=torch.float32), 
            k.contiguous().to(dtype=torch.float32),
            v.contiguous().to(dtype=torch.float32),
        )
        y = v 
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


def fla_parallel_based_test(dt, q, k, v, d, verbose=True, **kwargs):
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y = parallel_based(q, k, v, False, False)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot

def mamba2_kernel_test(dt, q, k, v, d, verbose=True, **kwargs):

    b, h, n, d = q.shape
    dv = v.shape[-1]

    try:
        # Set so Mamba-2's (2 * dim * dstate) is ~ Based's (dim * (1 + 16 + 16^2))
        import torch.nn.functional as F
        dt = F.softplus(torch.randn(b, n, h, dtype=torch.float32, device=q.device) - 4)
        A = (-torch.exp(torch.rand(h, dtype=torch.float32, device=q.device)))
        k = torch.randn(b, n, 1, 256, dtype=torch.bfloat16, device=q.device)
        q = torch.randn(b, n, 1, 256, dtype=torch.bfloat16, device=q.device)    
        v = torch.randn(b, n, h, dv*2, dtype=torch.bfloat16, device=q.device)

        torch.cuda.synchronize()
        t0 = time.time()

        y = mamba_chunk_scan_combined(v, dt, A, k, q, chunk_size=64, D=None)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot
    
def mamba_kernel_test(dt, q, k, v, d, verbose=True, **kwargs):

    b, h, n, d = q.shape
    dv = v.shape[-1]
    dstate = 16 # from Mamba paper, note state is 8x smaller then Based

    try:
        dmodel = dv*h*2
        A = torch.randn(dmodel, dstate, dtype=torch.float32, device=q.device)
        x = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=q.device)
        dt = torch.randn(b, dmodel,n, dtype=torch.bfloat16, device=q.device)    
        B = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=q.device)
        C = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=q.device)
        D = torch.randn(dmodel, dtype=torch.bfloat16, device=q.device)
        z = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=q.device)
        dt_proj_bias = torch.randn(dmodel, dtype=torch.bfloat16, device=q.device)

        torch.cuda.synchronize()
        t0 = time.time()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot

def based_kernel_test(dt, Q, K, V, d, verbose=True, device='4090'):
    o   = torch.zeros_like(V)
    b, h, n, d = Q.shape 
    dv = V.shape[-1]

    kv_state_a2 = torch.empty((b, h, d*d, dv), dtype=torch.bfloat16, device='cuda')
    kv_state_a1 = torch.empty((b, h, dv, d), dtype=torch.bfloat16, device='cuda')
    kv_state_a0 = torch.empty((b, h, d), dtype=torch.bfloat16, device='cuda')
    add_scale, output_state = int(1), int(0)

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        tk.based_linear_prefill(
            add_scale,output_state,
            Q, K, V, o, 
            kv_state_a2,kv_state_a1,kv_state_a0
        )

        torch.cuda.synchronize()
        t1 = time.time()
        o += torch.zeros_like(o) # trigger an error if one exists
        tot = t1-t0
    except Exception as e:
        # print(f"Error: {e}")
        tot = -1
        o = None

    return o, tot


def linear_attn_forward_benchmark_batch(dt, methods, verbose=False, use_ones=False, profile=False):
    num_iters = 10

    method2timing = defaultdict(dict)
    for b in [1, 2, 4, 8, 16, 32, 128, 256]:
        h = 16
        n = 8192 
        d = 16
        dv = 64
        print(f"{b=}, {n=}, {d=}, {h=}")

        for name, fn, in methods.items():
            if "mamba" in name and (b*h*dv*256) > 500000000: continue

            try:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

                print(f"Running {name}...")
                lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
                lst_time = [x[-1] for x in lst]
                _time = median(lst_time)
                if b > 1 and _time > 0: method2timing[name][b] = _time * 1000
            
                torch.cuda.empty_cache()
            except:
                pass 

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    plt.savefig(f'plots/benchmark-lin-attn-fwd-L{n}.png', bbox_inches='tight')


def linear_attn_forward_benchmark_seqlen(dt, methods, verbose=False, use_ones=False, profile=False):
    num_iters = 10
    method2timing = defaultdict(dict)
    for n in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        b = 16
        h = 16
        d = 16
        dv = 64
        print(f"{b=}, {n=}, {d=}, {h=}")

        for name, fn, in methods.items():
            if "mamba" in name and (b*h*dv*256) > 100000000: continue

            try:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

                print(f"Running {name}...")
                lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
                lst_time = [x[-1] for x in lst]
                _time = median(lst_time)
                if n > 256 and _time > 0: method2timing[name][n] = _time * 1000
                torch.cuda.empty_cache()
            except:
                pass

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    plt.savefig(f'plots/benchmark-lin-attn-fwd-B{b}.png', bbox_inches='tight') 


############## Efficiency Measurements #############

def get_flops(batch, seqlen, headdim, nheads, featuredim):
    expanded_dim = featuredim * featuredim + featuredim + 1
    f = 2 * batch * seqlen * nheads * expanded_dim # compute feature map on q and k
    f += batch * seqlen * nheads * headdim * expanded_dim # (k * v)
    f += batch * seqlen * nheads * headdim * expanded_dim # (cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1)
    return f


#  Function to calculate the efficiency in teraflops
def efficiency(flops, time):
    # Convert flop to teraflops and time to milliseconds
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms


def measure_efficiency(dt, methods, verbose=False, use_ones=False, profile=False):
    num_iters = 10
    method2timing = defaultdict(dict)
    
    n = 8192
    b = 64
    h = 32
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    for method_name, fn in methods.items():
        if "based" not in method_name.lower(): continue
        try:
            Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
            K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
            V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

            lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
            lst_time = [x[-1] for x in lst][2:]
            _time = median(lst_time)
            if _time < 0: continue

            flops = get_flops(b, n, dv, h, d)
            microseconds = _time * 1000000
            eff = efficiency(flops, microseconds)
            print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS")
            torch.cuda.empty_cache()  
        except:
            pass


if __name__ == "__main__":
    print(f"Storing results at 'results/'")
    os.makedirs('plots', exist_ok=True)

    print("Benchmarking the kernels...")
    methods = {
            # Based
            'Based PyTorch': pytorch_test,
            # 'Based Fast Transformers': fast_transformer_test, 
            'Based TK': based_kernel_test,

            # Triton Based
            'Based Flash Linear Attention': fla_parallel_based_test,

            # FA2
            'Flash Attention': flash_attention_test,

            # Mamba-2
            "Mamba": mamba_kernel_test,
            "Mamba-2": mamba2_kernel_test,
        }

    linear_attn_forward_benchmark_batch(torch.bfloat16, methods, verbose=False)
    linear_attn_forward_benchmark_seqlen(torch.bfloat16, methods, verbose=False)
    measure_efficiency(torch.bfloat16, methods, verbose=False, use_ones=False, profile=False)
