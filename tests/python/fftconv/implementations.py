import torch
import sys
import os
import time
import argparse
import math

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


try:
    from flashfftconv import FlashFFTConv
    print(f"Successfully imported FlashFFTConv")
except:
    FlashFFTConv = None
    print("Could not import FlashFFTConv. Please clone and install.")

try:
    from fftconv.baselines.tk_fftconv import TKFFTConv
    from fftconv.baselines.tk_fftconv import ref_fftconv
    print(f"Successfully imported TKFFTConv")
except:
    TKFFTConv = None
    print("Could not import TKFFTConv. Please install from TK.")


################ FFT Convolution ################


def get_flops(batch, seqlen, headdim, nheads):
    d_model = headdim * nheads
    flops = 2 * (10 * seqlen * math.log(seqlen, 2) * d_model * batch)
    return flops 


def get_fft_inputs(dt, b, h, n, dv):
    d_model = (h * dv)
    u = torch.randn((b, d_model, n), dtype=dt).to('cuda')
    k = torch.randn((d_model, n), dtype=torch.float32).to('cuda')
    return u, k


def fftconv_tk_test(dt, b, h, n, dv, verbose=True, **kwargs):
    if n not in [1024, 4096]:
        print(f"Sequence length {n} not supported by TKFFTConv")
        return None, -1
    
    x, k = get_fft_inputs(dt, b, h, n, dv)
    tk_fft = TKFFTConv(k, seqlen=n, H=(h*dv), dtype=x.dtype).to(x.device)

    torch.cuda.synchronize()
    t0 = time.time()

    y_tk = tk_fft(x, k)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    assert not np.isnan(y_tk.float().cpu()).any(), "NaN values detected in output 'y_tk'"
    assert not np.isinf(y_tk.float().cpu()).any(), "Inf values detected in output 'y_tk'"

    y_ref = ref_fftconv(x, k, n)
    max_diff = torch.abs(y_tk - y_ref).max().item()
    mean_diff = torch.abs(y_tk - y_ref).mean().item()
    print(f"tk-torch {max_diff=}, {mean_diff=}")

    return y_tk, tot


def fftconv_cutlass_test(dt, b, h, n, dv, verbose=True, **kwargs):
    if n not in [1024, 4096]:
        print(f"Sequence length {n} not supported by TKFFTConv")
        return None, -1
    
    u, k = get_fft_inputs(dt, b, h, n, dv)
    conv_flashfft = FlashFFTConv(n, dtype=u.dtype).to(u.device)

    torch.cuda.synchronize()
    t0 = time.time()

    y_flashfft = conv_flashfft(u, k)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    return y_flashfft, tot


def fftconv_pytorch_test(dt, b, h, n, dv, verbose=True, **kwargs):
    if n not in [1024, 4096]:
        print(f"Sequence length {n} not supported by TKFFTConv")
        return None, -1
    
    x, k = get_fft_inputs(dt, b, h, n, dv)

    torch.cuda.synchronize()
    t0 = time.time()

    x = ref_fftconv(x, k, n)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    return x, tot

IMPLEMENTATIONS = {
    "pytorch_conv": fftconv_pytorch_test,
    "tk_fftconv": fftconv_tk_test,
    "flashfft_conv": fftconv_cutlass_test
}

