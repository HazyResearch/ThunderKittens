import torch
import sys
import os
import time
import argparse
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import warnings
warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")


from utils import efficiency

import attention.implementations as attention
import hedgehog.implementations as hedgehog
import based.implementations as based
import rotary.implementations as rotary
# import mamba2.implementations as mamba2
# import fftconv.implementations as fftconv
# import layernorm.implementations as layernorm



############## Efficiency Measurements #############

def measure_efficiency(dt, n, method_name, method, verbose=False):
    num_iters = 10
    n_warmup = 2
    method2timing = defaultdict(dict)

    b = 16
    h = 16
    dv = 64
    if verbose:
        print(f"{b=}, {n=}, {h=}, {dv=}")

    if 'c=t' in method_name:
        causal = True
        flops = mod.get_flops(b, n, dv, h, causal=causal)
    elif 'c=f' in method_name:
        causal = False
        flops = mod.get_flops(b, n, dv, h, causal=causal)
    else:
        flops = mod.get_flops(b, n, dv, h)  

    # try:
    if 1:
        lst = [method(dt, b, h, n, dv, verbose=verbose) for _ in range(num_iters)]
        lst_time = [x[-1] for x in lst][n_warmup : ] # skip the first two iterations (warmup)
        _time = median(lst_time)
    # except:
    #     if verbose:
    #         print(f"Error: {sys.exc_info()[0]}")
    #     _time = -1

    microseconds = _time * 1000000
    eff = efficiency(flops, microseconds)
    if verbose:
        print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, Time: {_time:.4f} s and FLOPS: {flops:.2f}")
    torch.cuda.empty_cache()
    return eff, _time

if __name__ == "__main__":
    print("Benchmarking the kernels...")
    print("============" * 8)

    verbose = False

    for mod in [
        # based, 
        # attention, 
        # rotary,
        hedgehog, 
        # fftconv, 
        # layernorm, 
        # mamba2, 
    ]:
        implementations_list = []
        implementations_fwd = mod.IMPLEMENTATIONS
        implementations_list.append(implementations_fwd)

        try:
            implementations_bwd = mod.IMPLEMENTATIONS_BWD
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
                    # 1024, 
                    2048, 
                    # 4096, 
                    # 8192, 
                    # 16384
                ]:
                    if "conv" in m and n not in [1024, 4096]:
                        continue
                    if verbose:
                        print(f"Sequence Length: {n}")
                    tflops, timing = measure_efficiency(torch.bfloat16, n, m, method, verbose=verbose)
                    if tflops > 0: 
                        flops_result[n] = tflops
                        timing_result[n] = timing
                method2tflops[m] = flops_result
                method2timing[m] = timing_result

            # print table pretty
            import pandas as pd
            df = pd.DataFrame(method2tflops).replace(np.nan, 'OOM', regex=True)
            print(df)
            print("============" * 8)
    