


import sys
import os
import time
import argparse
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
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


from implementations import TaylorExp, TERMS, eps, based_test, get_based_inputs


def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol


def test_correctness(): 
    dt = torch.bfloat16
    b = 2
    n = 2048
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    head_idx = torch.randint(0, h, (1,)).item()
    batch_idx = torch.randint(0, b, (1,)).item()

    Q, K, V = get_based_inputs(b, h, n, dv, dt)

    # get outputs from different methods
    pytorch_test = partial(based_test, causal=True, is_forwards=True, method_str="pytorch")
    tk_test = partial(based_test, causal=True, is_forwards=True, method_str="tk")
    pytorch_v1, _  = pytorch_test(dt, b, h, n, dv)
    tk_outputs, _  = tk_test(dt, b, h, n, dv)
    tk_outputs, kv_state_tk = outputs[0], outputs[1]

    torch.set_printoptions(sci_mode=False)

    # check overall outputs
    print(f"Note we find numerical differences upon inspecting the tensor outputs:\n")
    print(f"Checking outputs:")
    __eq("PyTorch v1 - Based TK", pytorch_v1, tk_outputs, debug=False)

    print(pytorch_v1[1,head_idx,70:72,:4])
    print(tk_outputs[1,head_idx,70:72,:4])
    print()

    print(pytorch_v1[0,head_idx,128:130,:4])
    print(tk_outputs[0,head_idx,128:130,:4])
    print()

    print(pytorch_v1[1,head_idx,500:502,:4])
    print(tk_outputs[1,head_idx,500:502,:4])
    print()

    print("---"*10)

    # kv states
    __eq("PyTorch v1 - TK", kv_state_v1, kv_state_tk, debug=False)
    print(kv_state_v1[1,head_idx,0,:4])
    print(kv_state_tk[1,head_idx,0,:4])
    max_diff = torch.max(torch.abs(kv_state_v1 - kv_state_tk))
    pos_max_diff = int(torch.argmax(torch.abs(kv_state_v1 - kv_state_tk)))
    tk_val = kv_state_tk.cpu().float().numpy().flatten()[pos_max_diff]
    pytorch_val = kv_state_v1.cpu().float().numpy().flatten()[pos_max_diff]
    print(f"Max diff: {max_diff}; {tk_val=}, {pytorch_val=}")
    print(f"Max at position: {pos_max_diff} out of {kv_state_v1.numel()} / {kv_state_tk.numel()}")

if __name__ == "__main__":
    test_correctness()

