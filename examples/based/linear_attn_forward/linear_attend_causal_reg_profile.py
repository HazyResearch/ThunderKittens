import torch
import sys
import os
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
import linear_attend_causal_reg as mod

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
from typing import Optional

# sys.path.append("/var/cr05_data/sim_data/code/based/train/")
# sys.path.append("/cudatastic/based-dev/train/")
# from csrc import causal_dot_product


def make_causal(X):
    (b,h,n,m) = X.shape
    mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
    X[mask] = 0.
    return X

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, **kwargs: any):
        super().__init__()
        self.r2  = 1 #math.sqrt(2)
        self.rd  = 2 #math.sqrt(input_dim)
        self.rrd = 1 #math.sqrt(self.rd)
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)


def pytorch_test(dt, Q, K, V, d, verbose=True):
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        O   = torch.einsum("bhnd,bhmd->bhnm", Q, K)**2
        O2  = make_causal(O)
        T2  = torch.einsum("bhnm,bhmd->bhnd", O2, V)
        T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
        T1 = torch.einsum("bhnm,bhme->bhne", T1a, V)  
        T0  = V.cumsum(dim=2)
        y  = T0 #+ T1 + T2/2

        torch.cuda.synchronize()
        t1 = time.time()
        peak_mem = torch.cuda.max_memory_allocated(device='cuda')
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}") # likely OOM
        tot = -1
        peak_mem=-1
        y= None

    return y, tot, peak_mem


def pytorch_test_v2(dt, Q, K, v, d, verbose=True):
    feature_map = TaylorExp(input_dim=d)

    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        q, k = feature_map(Q), feature_map(K)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2)  # causal 
        k_state = k.cumsum(dim=2)
        y = ((q * kv_state).sum(dim=-1)) #/ (q * k_state).sum(dim=-1))

        torch.cuda.synchronize()
        t1 = time.time()
        peak_mem = torch.cuda.max_memory_allocated(device='cuda')
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}") # likely OOM
        tot = -1
        peak_mem=-1
        y= None

    return y, tot, peak_mem


def fast_transformer_test(dt, q, k, v, d, verbose=True):

    feature_map = TaylorExp(input_dim=d)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    q, k = feature_map(q), feature_map(k)
    v = causal_dot_product(
        q.contiguous().to(dtype=torch.float32), 
        k.contiguous().to(dtype=torch.float32),
        v.contiguous().to(dtype=torch.float32),
    )
    # z = 1 / torch.einsum(
    #         "bhld,bhld->bhl", 
    #         q.to(dtype=torch.float32), 
    #         k.to(dtype=torch.float32).cumsum(2)
    #     )
    y = v # * z[..., None]

    torch.cuda.synchronize()
    t1 = time.time()
    peak_mem = torch.cuda.max_memory_allocated(device='cuda')
    tot = t1-t0
    return y, tot, peak_mem


def based_kernel_test(dt, Q, K, V, d, verbose=True):
    o   = torch.zeros_like(V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        mod.a012_compute(Q,K,V, o)
    except:
        breakpoint()

    torch.cuda.synchronize()
    t1 = time.time()
    peak_mem = torch.cuda.max_memory_allocated(device='cuda')
    tot = t1-t0
    return o, tot, peak_mem


def linear_attn_forward_benchmark(dt,verbose=False, use_ones=False, profile=False):
    num_iters = 10
    methods = {
        # 'Pure PyTorch (Alg. 1)': pytorch_test, 
        #'Pure PyTorch': pytorch_test_v2,
        #'Fast Transformers Kernel': fast_transformer_test, 
        'Based Kernel': based_kernel_test
    }
    method2timing = defaultdict(dict)
    method2mem = defaultdict(dict)
    for b in [1, 2, 4, 8, 16, 32, 128, 256]:
        h = 16
        n = 1024 
        d = 16
        dv = 64
        print(f"{b=}, {n=}, {d=}, {h=}")

        Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

        for name, fn, in methods.items():
            print(f"Running {name}...")
            lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
            lst_time = [x[1] for x in lst]
            lst_mem = [x[-1] for x in lst]
            _time = median(lst_time)
            _mem = median(lst_mem)

            if b > 1 and _time > 0: method2timing[name][b] = _time * 1000
            if b > 1 and _time > 0: method2mem[name][b] = _mem

    print("\nEfficiency vs. Batch Size:")
    print("Timings:")
    print(method2timing['Based Kernel'])
    print("Memory:")
    print(method2mem['Based Kernel'])

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('h100_lin-attn-fwd_benchmark.pdf', format='pdf', bbox_inches='tight')


def linear_attn_forward_benchmark_seqlen(dt,verbose=False, use_ones=False, profile=False):
    num_iters = 10
    methods = {
        # 'Pure PyTorch (Alg. 1)': pytorch_test, 
        #'Pure PyTorch': pytorch_test_v2,
        #'Fast Transformers Kernel': fast_transformer_test, 
        'Based Kernel': based_kernel_test
    }
    method2timing = defaultdict(dict)
    method2mem = defaultdict(dict)
    for n in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 131072]:
        b = 4
        h = 16
        d = 16
        dv = 64
        print(f"{b=}, {n=}, {d=}, {h=}")

        Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

        for name, fn, in methods.items():
            print(f"Running {name}...")
            lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
            lst_time = [x[1] for x in lst]
            lst_mem = [x[-1] for x in lst]
            _time = median(lst_time)
            _mem = median(lst_mem)

            if n > 256 and _time > 0: method2timing[name][n] = _time * 1000
            if n > 256 and _mem > 0: method2mem[name][n] = _mem

    print(f"\nEfficiency vs. Seqlen:")
    print(f"Timings:")
    print(method2timing['Based Kernel'])
    print(f"Memory:")
    print(method2mem['Based Kernel'])

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.legend()

    # save pdf
    plt.savefig('h100_lin-attn-fwd_benchmark_seqlen.pdf', format='pdf', bbox_inches='tight') 


def linear_attn_correct(dt):
    b = 4
    n = 1024
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    pytorch_test_result = pytorch_test(dt, Q, K, V, d)
    # pytorch_test_v2_result = pytorch_test_v2(dt, Q, K, V, d) 
    # fast_transformer_test_result = fast_transformer_test(dt, Q, K, V, d)
    based_kernel_test_result = based_kernel_test(dt, Q, K, V, d)

    # __eq("PyTorch Test v1 - PyTorch Test v2", pytorch_test_result[0], pytorch_test_v2_result[0], debug=False)
    # __eq("PyTorch Test v1 - Fast Transformer Test", pytorch_test_result[0], fast_transformer_test_result[0], debug=False)  # fp. accum. error
    __eq("PyTorch Test v1 - Based Kernel Test", pytorch_test_result[0], based_kernel_test_result[0], debug=False)

print("Benchmarking the kernels...")
# linear_attn_forward_benchmark(torch.bfloat16, verbose=False)
# linear_attn_forward_benchmark_seqlen(torch.bfloat16, verbose=False)

print("Correctness test...")
linear_attn_correct(torch.bfloat16)