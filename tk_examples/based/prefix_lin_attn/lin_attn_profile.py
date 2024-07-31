import torch
import sys
import os
import time
import argparse
import pandas as pd
from collections import defaultdict
from statistics import median
import torch.nn as nn
from einops import rearrange

# These installs are for pulling in the TK source
sys.path.append('../../')
sys.path.append('../../../')
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')

loaded_correctly = []

# To import the ThunderKittens based kernels
try:
    sys.path.append("../linear_attn_forward/H100")
    import lin_attn as mod
    print(f"Successfully imported TK based_H100 kernel")
except:
    loaded_correctly.append('Based TK')
    mod = None
    print("Could not import TK based_H100 kernel; Uncomment it in methods dict below if you don't want to evaluate it.")

# To import the ThunderKittens JRT kernels
try:
    sys.path.append("H100")
    import prefill as mod_jrt
    print(f"Successfully imported TK JRT_H100 kernel")
except:
    loaded_correctly.append('JRT TK')
    mod_jrt = None
    print("Could not import TK JRT_H100 kernel; Uncomment it in methods dict below if you don't want to evaluate it.")

# To compare to Faster Transformers
try:
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    loaded_correctly.append('Fast transformer')
    causal_dot_product = None
    print("Could not import causal_attention_cuda; Uncomment it in methods dict below if you don't want to evaluate it.")

# To compare to Flash Linear Attention 
try:
    # Install from https://github.com/sustcsonglin/flash-linear-attention
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    print(f"Successfully imported flash_linear_attention")
except:
    loaded_correctly.append('FLA')
    fused_chunk_based, parallel_based, naive_parallel_based = None, None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention; Uncomment it in methods dict below if you don't want to evaluate it.")

# To compare to Flash Attention
try:
    from flash_attn import flash_attn_func
    print(f"Successfully imported flash_attention")
except:
    loaded_correctly.append('FA')
    flash_attn_func = None
    print("Could not import flash_attention; Uncomment it in methods dict below if you don't want to evaluate it.")


assert not loaded_correctly, f"Please install the necessary packages to run the benchmarks or comment out the benchmarks that you don't want.  Missing: {loaded_correctly}"


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


################ JRT Versions ##################


def jrt_pytorch_test(dt, Q, K, V, K_enc, V_enc, d, verbose=True):
    feature_map = TaylorExp(input_dim=d)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        
        Q, K = feature_map(Q), feature_map(K)
        K_enc = feature_map(K_enc)

        # Standard attention
        A_qk = torch.einsum("bhnd,bhmd->bhnm", Q, K) 
        A_qk = torch.tril(A_qk)
        y1 = torch.einsum("bhnm,bhme->bhne", A_qk.to(Q.dtype), V.to(Q.dtype))

        # Cross attention
        A_qk_2 = torch.einsum("bhnd,bhmd->bhnm", Q, K_enc)
        y2 = torch.einsum("bhnm,bhme->bhne", A_qk_2.to(Q.dtype), V_enc.to(Q.dtype))

        # Encoder attention
        y = y1 + y2

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}") # likely OOM
        tot = -1
        y= None

    return y, tot


def jrt_based_kernel_test(dt, Q, K, V, K_enc, V_enc, d, verbose=True):
    o   = torch.zeros_like(V)

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        mod_jrt.jrt_prefill_tk(Q,K,V,K_enc,V_enc,o)
        o += torch.zeros_like(o) 

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        o = None

    return o, tot


################ Based Versions ################

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
    feature_map = TaylorExp(input_dim=d)
    try:
        torch.cuda.synchronize()
        t0 = time.time() 

        Q, K = feature_map(Q), feature_map(K)
        A_qk = torch.einsum("bhnd,bhmd->bhnm", Q, K) 
        A_qk = torch.tril(A_qk)
        y = torch.einsum("bhnm,bhme->bhne", A_qk.to(Q.dtype), V.to(Q.dtype))

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
    except Exception as e:
        print(f"Error: {e}")
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


def based_kernel_test(dt, Q, K, V, d, add_scale=False,output_state=False, verbose=True):
    b, h, n, d = Q.shape
    dv = V.shape[-1]
    o   = torch.zeros_like(V)
    kv_state_a2 = torch.zeros((b, h, dv, d*d), dtype=dt, device='cuda')
    kv_state_a1 = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    kv_state_a0 = torch.zeros((b, h, dv), dtype=dt, device='cuda')

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        mod.based_fwd_tk(int(add_scale),int(output_state),Q,K,V,o,kv_state_a2,kv_state_a1,kv_state_a0)

        o += torch.zeros_like(o) # trigger an error if one exists

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        print(f"Error: {e}")
        tot = -1
        o = None

    return o, tot


################### Benchmarking and Correctness Tests ####################


def linear_attn_forward_benchmark_batch(dt, methods, verbose=False, use_ones=False, profile=False):
    num_iters = 10

    method2timing = defaultdict(dict)
    for b in [1, 2, 4, 8, 16, 32, 64]:
        h = 16
        n = 16384 
        twice_n = n * 2
        half_n = n // 2
        d = 16
        dv = 64
        print(f"{b=}, {n=}, {d=}, {h=}")

        for name, fn, in methods.items():
            print(f"Running {name}...")
            if "JRT" not in name:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
                # free memory
                del Q, K, V
            elif "JRT-RNN" in name:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
                K_enc = torch.randn(b,h,half_n,d, dtype=dt, device='cuda')/d
                V_enc = torch.randn(b,h,half_n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q, K, V, K_enc, V_enc, d) for _ in range(num_iters)]
                # free memory
                del Q, K, V, K_enc, V_enc
            else:
                Q2   = torch.randn(b,h,twice_n,d, dtype=dt, device='cuda')/d
                K2   = torch.randn(b,h,twice_n,d, dtype=dt, device='cuda')/d
                V2   = torch.randn(b,h,twice_n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q2, K2, V2, d) for _ in range(num_iters)]
                # free memory
                del Q2, K2, V2
            lst_time = [x[-1] for x in lst]
            _time = median(lst_time)
            if b > 1 and _time > 0: method2timing[name][b] = _time * 1000

    df = pd.DataFrame(method2timing)
    print(df)


def linear_attn_forward_benchmark_seqlen(dt, methods, verbose=False, use_ones=False, profile=False):
    num_iters = 10
    method2timing = defaultdict(dict)
    for n in [2048, 4096, 8192, 16384, 32768]:
        b = 16
        h = 16
        d = 16
        dv = 64
        half_n = n // 2
        twice_n = n * 2
        print(f"{b=}, {n=}, {d=}, {h=}")

        for name, fn, in methods.items():
            print(f"Running {name}...")
            if "JRT" not in name:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q, K, V, d) for _ in range(num_iters)]
                # free memory
                del Q, K, V
            elif "JRT-RNN" in name:
                Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
                V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
                K_enc = torch.randn(b,h,half_n,d, dtype=dt, device='cuda')/d
                V_enc = torch.randn(b,h,half_n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q, K, V, K_enc, V_enc, d) for _ in range(num_iters)]
                # free memory
                del Q, K, V, K_enc, V_enc
            else:
                Q2   = torch.randn(b,h,twice_n,d, dtype=dt, device='cuda')/d
                K2   = torch.randn(b,h,twice_n,d, dtype=dt, device='cuda')/d
                V2   = torch.randn(b,h,twice_n,dv, dtype=dt, device='cuda')/dv
                lst = [fn(dt, Q2, K2, V2, d) for _ in range(num_iters)]
                # free memory
                del Q2, K2, V2

            lst_time = [x[-1] for x in lst]
            _time = median(lst_time)
            if n > 256 and _time > 0: method2timing[name][n] = _time * 1000

    df = pd.DataFrame(method2timing)
    print(df)


def linear_attn_correct(dt):
    b = 4
    n = 2048
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    pytorch_test_result = pytorch_test(dt, Q, K, V, d)
    based_kernel_test_result = based_kernel_test(dt, Q, K, V, d)
    __eq("PyTorch v1 - Based TK", pytorch_test_result[0], based_kernel_test_result[0], debug=False)


def jrt_linear_attn_correct(dt):
    b = 4
    n = 1024
    half_n = n // 2
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    K_enc = torch.randn(b,h,half_n,d, dtype=dt, device='cuda')/d
    V_enc = torch.randn(b,h,half_n,dv, dtype=dt, device='cuda')/dv

    jrt_torch_result = jrt_pytorch_test(dt, Q, K, V, K_enc, V_enc, d)
    jrt_tk_result = jrt_based_kernel_test(dt, Q, K, V,  K_enc, V_enc, d)
    __eq("PyTorch - JRT TK", jrt_torch_result[0], jrt_tk_result[0], debug=False)


############## Efficiency Measurements #############

def get_flops(batch, seqlen, headdim, nheads, featuredim):
    expanded_dim = featuredim * featuredim + featuredim + 1

    f = 2 * batch * seqlen * nheads * expanded_dim
    f += batch * seqlen * nheads * headdim * expanded_dim  # (k * v)
    f +=  batch * seqlen * nheads * headdim * expanded_dim # (cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim  # (q * (k * v))
    f += batch * seqlen * nheads * headdim * expanded_dim  # .sum(dim=-1)

    return f


#  Function to calculate the efficiency in teraflops
def efficiency(flops, time):
    # Convert flop to teraflops and time to milliseconds
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms


def measure_efficiency(dt, methods, device, verbose=False, use_ones=False, profile=False):
    num_iters = 100
    method2timing = defaultdict(dict)
    
    n = 2048
    b = 8
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    lst = [fast_transformer_test(dt, Q, K, V, d, device=device) for _ in range(num_iters)]
    lst_time = [x[-1] for x in lst]
    _time = median(lst_time)

    flops = get_flops(b, n, dv, h, d)
    microseconds = _time * 1000000
    eff = efficiency(flops, microseconds)

    print(f"Efficiency: {eff:.2f} TFLOPS")


if __name__ == "__main__":

    print(f"Storing results at 'results/'")
    os.makedirs('results', exist_ok=True)

    print("Benchmarking the kernels...")
    methods = {
            # Based
            'Based PyTorch': pytorch_test,
            'Based CUDA': based_kernel_test,
            'Based Triton': fla_parallel_based_test,

            # JRT
            'JRT-RNN PyTorch': jrt_pytorch_test,
            'JRT-Prompt CUDA': based_kernel_test,
            'JRT-RNN CUDA': jrt_based_kernel_test,

            # Other
            'Flash Attention 2': flash_attention_test,
            'Fast Transformer CUDA': fast_transformer_test, 
        }

    linear_attn_forward_benchmark_batch(torch.bfloat16, methods, verbose=False)
    linear_attn_forward_benchmark_seqlen(torch.bfloat16, methods, verbose=False)

    print("Correctness test...")
    linear_attn_correct(torch.bfloat16)
    jrt_linear_attn_correct(torch.bfloat16)


