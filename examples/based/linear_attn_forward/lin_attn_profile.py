import torch
import sys
import os
import time
import argparse

# These installs are for pulling in the TK source
sys.path.append('../../../')
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')


# To import the ThunderKittens based kernels
try:
    sys.path.append("4090")
    import lin_attn as mod
    print(f"Successfully imported TK based_4090 kernel")
except:
    mod = None
    print("Could not import TK based_4090 kernel")

try:
    sys.path.append("H100")
    import lin_attn_h100 as mod_h100
    print(f"Successfully imported TK based_H100 kernel")
except:
    mod_h100 = None
    print("Could not import TK based_H100 kernel")


from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn

# To compare to Faster Transformers
try:
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    causal_dot_product = None
    print("Could not import causal_attention_cuda")

# To compare to Flash Linear Attention 
try:
    # Install from https://github.com/sustcsonglin/flash-linear-attention
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    print(f"Successfully imported flash_linear_attention")
except:
    fused_chunk_based, parallel_based, naive_parallel_based = None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention")

# To compare to Flash Attention
try:
    from flash_attn import flash_attn_func
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention")


################ Flash Attention ################

def flash_attention_test(dt, q, k, v, d, verbose=True, **kwargs):
    q = torch.randn_like(v).transpose(1,2)
    k = torch.randn_like(v).transpose(1,2)
    v = torch.randn_like(v).transpose(1,2)

    try:
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
        print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


################ GLA Versions ################

def  fla_gla_chunk_test(dt, q, k, v, d, verbose=True, **kwargs):
    q = torch.randn_like(v)
    k = torch.randn_like(v)
    v = torch.randn_like(v)
    g = torch.randn_like(v)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        y, _ = chunk_gla(q, k, v, g)
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        y = None

    return y, tot

def fla_gla_fused_chunk_test(dt, q, k, v, d, verbose=True, **kwargs):
    q = torch.randn_like(v)
    k = torch.randn_like(v)
    v = torch.randn_like(v)
    g = torch.randn_like(v)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        y, _ = fused_chunk_gla(q, k, v, g)
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        y = None

    return y, tot

def fla_gla_fused_recurrent_test(dt, q, k, v, d, verbose=True, **kwargs):
    q = torch.randn_like(v)
    k = torch.randn_like(v)
    v = torch.randn_like(v)
    g = torch.randn_like(v)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        y, _ = fused_recurrent_gla(q, k, v, g)
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        y = None

    return y, tot

################ Based Versions ################

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


def pytorch_test(dt, Q, K, V, d, verbose=True, **kwargs):
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        O   = torch.einsum("bhnd,bhmd->bhnm", Q, K)**2
        O2  = make_causal(O)
        T2  = torch.einsum("bhnm,bhmd->bhnd", O2, V)
        T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
        T1 = torch.einsum("bhnm,bhme->bhne", T1a, V)  
        T0  = V.cumsum(dim=2)
        y  = T0 + T1 + T2/2

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}") # likely OOM
        tot = -1
        y= None

    return y, tot


def pytorch_test_v2(dt, Q, K, v, d, verbose=True, **kwargs):
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
        print(f"Error: {e}") # likely OOM
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
    except:
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
        print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


def fla_naive_parallel_based(dt, q, k, v, d, verbose=True, **kwargs):
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y = naive_parallel_based(q, k, v, False, False)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        y = None

    return y, tot


def fla_fused_chunk_test(dt, q, k, v, d, verbose=True, **kwargs):
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y = fused_chunk_based(q, k, v, False, False)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        y = None
    return y, tot


def based_kernel_test(dt, Q, K, V, d, verbose=True, device='4090'):
    o   = torch.zeros_like(V)

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        if device == '4090': 
            mod.based_fwd_tk(Q,K,V, o)
        elif device == 'H100':
            mod_h100.based_fwd_tk(Q,K,V, o)

        torch.cuda.synchronize()
        o += torch.zeros_like(o) # trigger an error if one exists
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        o = None

    return o, tot


def linear_attn_forward_benchmark_batch(dt, methods, device, verbose=False, use_ones=False, profile=False):
    num_iters = 10

    method2timing = defaultdict(dict)
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
            lst = [fn(dt, Q, K, V, d, device=device) for _ in range(num_iters)]
            lst_time = [x[-1] for x in lst]
            _time = median(lst_time)
            if b > 1 and _time > 0: method2timing[name][b] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig(f'results/{device}-lin-attn-fwd_benchmark-L{n}.pdf', format='pdf', bbox_inches='tight')


def linear_attn_forward_benchmark_seqlen(dt, methods, device, verbose=False, use_ones=False, profile=False):
    num_iters = 10
    method2timing = defaultdict(dict)
    for n in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
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
            lst = [fn(dt, Q, K, V, d, device=device) for _ in range(num_iters)]
            lst_time = [x[-1] for x in lst]
            _time = median(lst_time)
            if n > 256 and _time > 0: method2timing[name][n] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.legend()

    # save pdf
    plt.savefig(f'results/{device}-lin-attn-fwd_benchmark_seqlen-B{b}.pdf', format='pdf', bbox_inches='tight') 


def linear_attn_correct(dt, device):
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
    pytorch_test_v2_result = pytorch_test_v2(dt, Q, K, V, d) 
    based_kernel_test_result = based_kernel_test(dt, Q, K, V, d, device=device)

    print(f"Note we find numerical differences upon inspecting the tensor outputs:")
    __eq("PyTorch v1 - PyTorch v2", pytorch_test_result[0], pytorch_test_v2_result[0], debug=False)
    __eq("PyTorch - Based TK", pytorch_test_v2_result[0], based_kernel_test_result[0], debug=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Linear Attention Forward Benchmark')
    parser.add_argument('--device', type=str, default='4090', help='Device to benchmark on')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert args.device in ['4090', 'H100'], "Invalid device"

    print(f"Storing results at 'results/'")
    os.makedirs('results', exist_ok=True)

    print("Benchmarking the kernels...")
    methods = {
            # Based
            'Based PyTorch': pytorch_test_v2,
            'Based Fast Transformers': fast_transformer_test, 
            'Based Custom': based_kernel_test,

            # Triton Based
            # 'Based Fla Fused Chunk': fla_fused_chunk_test,
            # 'Based Fla Parallel': fla_parallel_based_test,

            # Triton Gated Linear Attention
            # 'GLA Chunk': fla_gla_chunk_test,
            # 'GLA Fused Chunk': fla_gla_fused_chunk_test,
            # 'GLA Fused Recurrent': fla_gla_fused_recurrent_test,

            # FA2
            'Flash Attention': flash_attention_test,
        }
    linear_attn_forward_benchmark_batch(torch.bfloat16, methods, args.device, verbose=False)
    linear_attn_forward_benchmark_seqlen(torch.bfloat16, methods, args.device, verbose=False)

    print("Correctness test...")
    linear_attn_correct(torch.bfloat16, args.device)


