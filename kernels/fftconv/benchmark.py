# Flash FFT Conv benchmark
# pip install git+https://github.com/HazyResearch/flash-fft-conv.git

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    from _C import fftconv
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from flashfftconv import FlashFFTConv
    print(f"Successfully imported FlashFFTConv")
except:
    FlashFFTConv = None
    print("Could not import FlashFFTConv. Please clone and install.")

try:
    from baselines.tk_fftconv import TKFFTConv
    from baselines.tk_fftconv import ref_fftconv
    print(f"Successfully imported TKFFTConv")
except:
    TKFFTConv = None
    print("Could not import TKFFTConv. Please install from TK.")


################ FFT Convolution ################


def get_flops(b: int, n: int, dv: int, h: int, causal: bool = False):
    flops = 2 * (10 * n * math.log(n, 2) * (h * dv) * b)
    return flops 


def get_fft_inputs(dt, b, h, n, dv):
    d_model = (h * dv)
    u = torch.randn((b, d_model, n), dtype=dt).to('cuda')
    k = torch.randn((d_model, n), dtype=torch.float32).to('cuda')
    return u, k

class PytorchFFTConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, k, n):
        return ref_fftconv(x, k, n)

def fftconv_tk_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):
    
    pytorch_method = PytorchFFTConv()
    if torch_compile and method_str == "conv_torch":
        print("Torchinductor does not support code generation for complex operators. Performance may be worse than eager.")
        print("Skipping torch compile for this method.")
        # try:
        #     pytorch_method = torch.compile(pytorch_method)
        # except Exception as e:
        #     print(f"Could not compile pytorch_method: {e}")
            
    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            x, k = get_fft_inputs(dt, b, h, n, dv)
            
            try:

                if method_str == "conv_torch":
                    torch.cuda.synchronize()
                    start_events[i].record() 
                    y = pytorch_method(x, k, n)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "conv_tk":
                    tk_fft = TKFFTConv(k, seqlen=n, H=(h*dv), dtype=x.dtype).to(x.device)
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y = tk_fft(x, k)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "conv_flashfft":
                    conv_flashfft = FlashFFTConv(n, dtype=u.dtype).to(u.device)
                    torch.cuda.synchronize()
                    start_events[i].record() 
                    with torch.no_grad():
                        y = conv_flashfft(x, k)
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

            except Exception as e:
                return None, -1

            torch.cuda.empty_cache()

    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    assert not np.isnan(y.float().cpu()).any(), "NaN values detected in output 'y'"
    assert not np.isinf(y.float().cpu()).any(), "Inf values detected in output 'y'"
    return y, tot


IMPLEMENTATIONS = {
    "conv_torch": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_torch"),
    "conv_tk": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_tk"),
    "conv_flashfft": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_flashfft"),
}

NAME = "FFTCONV"

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
