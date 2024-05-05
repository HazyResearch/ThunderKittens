import torch 
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-3.10')
import h100_train as mod

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def pytorch_test(Q, K, V, dO):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    o = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    
    o.backward(dO, retain_graph=True)
    
    qg = Q.grad
    kg = K.grad
    vg = V.grad
    
    return o, qg, kg, vg

def h100_fwd_kernel_test(Q, K, V, dO, verbose=True):
    o = torch.zeros_like(Q)
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=Q.dtype)
    mod.fwd_train_attend_ker_tk(Q, K, V, o, l_vec)
    
    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=Q.dtype)
    qg = torch.zeros_like(Q)
    kg = torch.zeros_like(K)
    vg = torch.zeros_like(V)
    
    mod.bwd_train_attend_ker_tk(Q, K, V, o, l_vec, d_vec, dO, qg, kg, vg)
    
    return o, qg, kg, vg

def h100_fwd_correct(): 
    b = 16
    h = 16
    n = 8192 * 2
    d = 64
    
    print(f"b={b} h={h} n={n} d={d}")
    
    Q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').requires_grad_()
    K = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').requires_grad_()
    V = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').requires_grad_()
    dO = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').requires_grad_()
    
    Q_tk = Q.clone().detach()
    K_tk = K.clone().detach()
    V_tk = V.clone().detach()
    dO_tk = dO.clone().detach()
    
    pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO)
    tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q_tk, K_tk, V_tk, dO_tk)
    
    # o diff
    # avg magnitude
    print(f"pt_o avg magnitude: {torch.mean(torch.abs(pt_o)).item()}")
    # avg diff (rmse)
    print(f"pt_o avg diff: {torch.mean(torch.abs(pt_o - tk_o)).item()}")
    
    # qg diff
    # avg magnitude
    print(f"pt_qg avg magnitude: {torch.mean(torch.abs(pt_qg)).item()}")
    # avg diff (rmse)
    print(f"pt_qg avg diff: {torch.mean(torch.abs(pt_qg - tk_qg)).item()}")
    
    # kg diff
    # avg magnitude
    print(f"pt_kg avg magnitude: {torch.mean(torch.abs(pt_kg)).item()}")
    # avg diff (rmse)
    print(f"pt_kg avg diff: {torch.mean(torch.abs(pt_kg - tk_kg)).item()}")
    
    # vg diff
    # avg magnitude
    print(f"pt_vg avg magnitude: {torch.mean(torch.abs(pt_vg)).item()}")
    # avg diff (rmse)
    print(f"pt_vg avg diff: {torch.mean(torch.abs(pt_vg - tk_vg)).item()}")
    
    
    __eq("h100_fwd", pt_o[0], tk_o[0], debug=False)
    __eq("h100_fwd_qg", pt_qg[0], tk_qg[0], debug=False)
    __eq("h100_fwd_kg", pt_kg[0], tk_kg[0], debug=False)
    __eq("h100_fwd_vg", pt_vg[0], tk_vg[0], debug=False)
    
print("Correctness test... ")
h100_fwd_correct()