


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

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")


from implementations import TaylorExp, TERMS, eps, based_kernel_test, get_based_inputs


def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol


def pytorch_test_v1(dt, Q, K, V, d, verbose=True, **kwargs):
    # SA: note the torch.float32 conversions are very important 

    b, h, n, D = Q.shape
    rd = math.sqrt(D) 
    rrd = math.sqrt(rd) 
    r2 = math.sqrt(2) 

    print(f"{b=}, {h=}, {n=}, {D=}")
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    # Overall output
    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))
    O2  = make_causal(O.to(torch.float32)**2)
    O1 = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2.to(torch.float32), V.to(torch.float32))
    T2 = T2/(r2 * r2 * rd * rd)
    T1 = torch.einsum("bhnm,bhmd->bhnd", O1.to(torch.float32), V.to(torch.float32))  
    T1 = T1/rd
    T0  = V.to(torch.float32).cumsum(dim=2)

    # KV states by term (a2)
    A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K.to(torch.float32),V.to(torch.float32),K.to(torch.float32)).cumsum(dim=2) / (r2 * rd) 
    A2 = A2[:, :, -1]
    A2 = rearrange(A2, 'b h e f d -> b h (e f) d')
    K2 = torch.einsum("bhnd,bhne->bhnde", K.to(torch.float32), K.to(torch.float32)) / (rd * r2)
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q.to(torch.float32), Q.to(torch.float32)) / (rd * r2)
    K2 = rearrange(K2, 'b h n d e  -> b h n ( d e )')
    Q2 = rearrange(Q2, 'b h n d e  -> b h n ( d e ) ')
    k_state_a2 = K2.to(torch.float32).cumsum(dim=2)
    D2 = torch.einsum("bhnd,bhnd->bhn", Q2.to(torch.float32), k_state_a2)

    # KV states by term (a1)
    A1 = torch.einsum("bhnd,bhne->bhnde",K.to(torch.float32),V.to(torch.float32)).cumsum(dim=2)  / rrd
    A1 = A1[:, :, -1].transpose(2, 3)
    k_state_a1 = K.to(torch.float32).cumsum(dim=2)/ (rrd)
    D1 = torch.einsum("bhnd,bhnd->bhn", Q.to(torch.float32), k_state_a1) / rrd 

    # KV states by term (a0)
    A0 = V.to(torch.float32).cumsum(dim=2)[:, :, -1]
    K0 = torch.ones(Q[..., :1].shape).to(Q.device)
    D0 =  K0.to(torch.float32).cumsum(dim=2).squeeze(-1)

    numerators   = [T0, T1, T2] 
    denominators = [D0, D1, D2]
    numerator   = sum([n for i, n in enumerate(numerators)   if i in TERMS]) 
    denominator = sum([n for i, n in enumerate(denominators) if i in TERMS]) 
    y = numerator

    kv_state = torch.cat([A0.unsqueeze(-1).transpose(2,3), A1.transpose(2,3), A2], dim=2)

    return y, kv_state


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
    pytorch_v1, kv_state_v1  = pytorch_test_v1(dt, Q, K, V, d)
    outputs, _  = based_kernel_test(dt, b, h, n, dv)
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