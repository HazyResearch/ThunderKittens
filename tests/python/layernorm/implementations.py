import torch
import sys
import os
import time
import argparse

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
    from baselines.layer_norm_triton import layer_norm_fn, RMSNorm
    print(f"Successfully imported layer_norm_fn")
except:
    layer_norm_fn = None
    print("Could not import layer_norm_fn. Please obtain from FlashAttention repo.")


################## Layer norm ######################

def get_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*headdim # compute the mask for dropout 
    f += batch*seqlen*nheads*headdim # add dropout and residual
    f += batch*seqlen*nheads*headdim # compute the mean
    f += batch*seqlen*nheads*headdim # compute the variance
    f += batch*seqlen*nheads*headdim # subtract mean
    f += batch*seqlen*nheads*headdim # divide by variance
    f += batch*seqlen*nheads*headdim # multiply by norm weight 
    f += batch*seqlen*nheads*headdim # add norm bias
    return f


def get_layer_norm_inputs(b, h, n, dv, dt):
    d_model = h * dv
    p = 0.1

    norm = nn.LayerNorm(d_model).cuda()
    dropout = nn.Dropout(p)

    x = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    residual = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)
    return x, residual, norm_weight, norm_bias, out, out_resid, norm, dropout


def pytorch_layernorm_test(dt, b, h, n, dv, verbose=True, **kwargs):
    (
        x, residual, norm_weight, norm_bias, 
        out, out_resid, norm, dropout 
    ) = get_layer_norm_inputs(b, h, n, dv, dt)
    rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, )
    residual_in_fp32 = True
    dtype = dt
    dropout_p = dropout.p
    
    torch.cuda.synchronize()
    t0 = time.time()

    dropped = dropout(x) 
    residual = (residual + dropped ) if residual is not None else dropped
    out = norm(residual.to(dtype=norm.weight.dtype))
    residual = residual.to(torch.float32) 

    t1 = time.time()
    tot = t1-t0
    return out, tot


def triton_layer_norm_test(dt, b, h, n, dv, verbose=True, **kwargs):
    (
        x, residual, norm_weight, norm_bias, 
        out, out_resid, norm, dropout 
    ) = get_layer_norm_inputs(b, h, n, dv, dt)
    rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, )
    residual_in_fp32 = True
    
    torch.cuda.synchronize()
    t0 = time.time()
    out, residual = layer_norm_fn(
        x,
        norm.weight,
        norm.bias,
        residual=residual,
        eps=norm.eps,
        dropout_p=dropout.p,
        rowscale=rowscale,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=False
    )
    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0
    return out, tot


def layer_norm_test(dt, b, h, n, dv, verbose=True, **kwargs):
    (
        x, residual, norm_weight, norm_bias, 
        out, out_resid, norm, dropout 
    ) = get_layer_norm_inputs(b, h, n, dv, dt)
    has_residual = int(residual is not None)
    
    import sys
    sys.path.append("/home/bfs/simran/clean2/ThunderKittens/examples/layer_norm/kernel/")
    import layer_norm as mod

    torch.cuda.synchronize()
    t0 = time.time()
    # tk.fused_layernorm(
    #     int(has_residual), float(dropout.p),
    #     x, residual, 
    #     norm_weight, norm_bias, 
    #     out, out_resid
    # )
    mod.fused_ln_tk(
            1, 0.1,
            x, residual, 
            norm_weight, norm_bias, 
            out, out_resid
    )
    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0
    return out, tot


