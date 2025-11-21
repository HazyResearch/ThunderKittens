import torch 
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from functools import partial

try:
    from mamba2.baselines.ssd_minimal import ssd_minimal_discrete 
except:
    from baselines.ssd_minimal import ssd_minimal_discrete

try:
    from _C import mamba2
    print(f"Successfully imported thunderkittens")
except ImportError:
    print(f'Failed to import thunderkittens; Please "pip setup.py install".')
    tk = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except ImportError:
    print(f"Failed to import mamba_chunk_scan_combined; Please 'pip install mamba_ssm'.")
    mamba_chunk_scan_combined = None


################### Mamba 2 ####################


def get_flops(b: int, n: int, dv: int, h: int, causal: bool = False) -> int:
    """Calculate FLOPs for Mamba-2 operation.
    
    Args:
        b: Batch size
        n: Sequence length
        dv: Head dimension
        h: Number of heads
        causal: Whether using causal attention
    
    Returns:
        Total number of FLOPs
    """
    # Constants
    chunk_length = 64
    state_dimension = 64
    ngroups = 1
    
    num_chunks = n // chunk_length
    seqlen = n
    nheads = h
    head_dimension = dv
    
    flops = 0
    
    # Mask and decay computations
    flops += seqlen * nheads  # cumsum
    flops += seqlen * nheads * chunk_length  # segsum
    flops += seqlen * nheads * chunk_length  # exp
    
    # Center blocks
    flops += 2 * num_chunks * ngroups * (chunk_length * chunk_length * state_dimension)  # QK^T
    flops += num_chunks * nheads * (chunk_length * chunk_length)  # L mask
    flops += 2 * num_chunks * nheads * (chunk_length * chunk_length * head_dimension)  # V mult
    
    # Low-rank factors
    flops += num_chunks * nheads * chunk_length * 2  # decay states
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)  # state computation
    
    # Inter-chunk operations
    flops += nheads * num_chunks * num_chunks * 2  # chunk decay
    flops += num_chunks * nheads * head_dimension * state_dimension  # state update
    
    # Output conversion
    flops += nheads * num_chunks * chunk_length  # decay out
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)  # final output
    flops += num_chunks * nheads * chunk_length * head_dimension  # state multiply
    
    # Final addition
    flops += num_chunks * chunk_length * nheads * head_dimension
    
    return b * flops


def get_inputs_mamba(dtype: torch.dtype, b: int, h: int, n: int, dv: int, verbose: bool = True):
    """Generate inputs for Mamba-2 computation.
    
    Args:
        dtype: Data type for tensors
        b: Batch size
        h: Number of heads
        n: Sequence length
        dv: Head dimension
        verbose: Whether to print dimension info
    
    Returns:
        Tuple of input tensors and parameters
    """
    torch.manual_seed(42)
    
    batch, seqlen = b, n
    chunk_size = 64
    dim = h * dv
    headdim = dv
    ngroups = 1
    dstate = 64
    device = "cuda"
    
    if verbose:
        print(f"Dimensions: batch={batch}, seqlen={seqlen}, heads={h}, dim={dim}, headdim={headdim}")
    
    x = torch.randn(batch, seqlen, h, headdim, dtype=torch.float32, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, h, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = -torch.exp(torch.rand(h, dtype=torch.float32, device=device)).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(h, dtype=torch.float32, device=device)
    
    return x, dt, A, B, C, D, chunk_size, torch.float32

class Mamba2Triton(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, dt, A, B, C, chunk_size, D=None):
        return mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)

def mamba2_test(dtype, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):
    
    triton_method = Mamba2Triton()
    if torch_compile and method_str == "mamba2_triton":
        try:
            triton_method = torch.compile(triton_method)
        except Exception as e:
            print(f"Could not compile triton_method: {e}")
            
    for stage in ['warmup', 'timed']:

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
        for i in range(num_iters):

            x, dt, A, B, C, D, chunk_size, _ = get_inputs_mamba(dtype, b, h, n, dv, verbose)
            q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
            k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
            v = rearrange(x*dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
            a = rearrange(A*dt, "b h l -> b l h").to(torch.float32).contiguous().requires_grad_()

            try:
                if method_str == "mamba2_tk": 
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = mamba2(q, k, v, a)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba2_triton":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = triton_method(x, dt, A, B, C, chunk_size, D=None)
                    end_events[i].record()
                    torch.cuda.synchronize()
                else:
                    assert 0, f"Unknown method: {method_str}"
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None, -1

            torch.cuda.empty_cache()

    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    return y, tot

    
IMPLEMENTATIONS = {
    "mamba2_triton": partial(mamba2_test, causal=True, is_forwards=True, method_str="mamba2_triton"),
    "mamba2_tk": partial(mamba2_test, causal=True, is_forwards=True, method_str="mamba2_tk"),
}

NAME = "MAMBA 2"

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
