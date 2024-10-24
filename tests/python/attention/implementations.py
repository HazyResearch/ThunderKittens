import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
import thunderkittens as tk

try:
    # To compare to Flash Attention 2
    from flash_attn import flash_attn_func as flash_attn_func2
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention. Please: pip install flash-attn")

try:
    # To compare to Flash Attention 3
    from flash_attn_interface import flash_attn_func as flash_attn_func3
    print(f"Successfully imported flash_attn_interface")
except:
    flash_attn_func3 = None
    print("Could not import flash_attn_interface. Please: clone flash-attention and install the hopper kernels")


def get_flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


################ ATTENTION ################

def get_attention_inputs(b, h, n, dv, dt=torch.bfloat16, pad_multiple=0, ):

    if pad_multiple:
        n = (n // pad_multiple) * pad_multiple + pad_multiple  

    q = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    k = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    v = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    dO = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    return q, k, v, dO


def pytorch_test(dt, b, h, n, dv, causal, is_forwards, verbose=True, **kwargs):
    q, k, v, dO = get_attention_inputs(b, h, n, dv, dt)
    
    try:

        torch.cuda.synchronize()
        t0 = time.time()

        QK = torch.matmul(q, k.transpose(-2, -1))
        QK /= (q.size(-1) ** 0.5)
        if causal:
            mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool).to(QK.device)
            QK.masked_fill_(mask, float('-inf'))
        QK = torch.nn.functional.softmax(QK, dim=-1)
        y = torch.matmul(QK, v)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None

    if is_forwards:
        return y, tot

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y.backward(dO)
    
        q_grad = q.grad
        k_grad = k.grad
        v_grad = v.grad
        
        q_grad = q_grad.to(torch.bfloat16)
        k_grad = k_grad.to(torch.bfloat16)
        v_grad = v_grad.to(torch.bfloat16)
        y = y.to(torch.bfloat16)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        q_grad = None
        k_grad = None
        v_grad = None
    
    return (q_grad, k_grad, v_grad), tot


def fa2_test(dt, b, h, n, dv, causal, is_forwards, verbose=True, **kwargs):
    q, k, v, dO = get_attention_inputs(b, h, n, dv, dt)
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    dO = dO.transpose(1,2)

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None

    if is_forwards:
        return y, tot

    try:

        torch.cuda.synchronize()
        t0 = time.time()

        y.backward(dO)

        q_grad = q.grad
        k_grad = k.grad
        v_grad = v.grad

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        q_grad = None
        k_grad = None
        v_grad = None

    return (q_grad, k_grad, v_grad), tot


def tk_test(dt, b, h, n, dv, causal, is_forwards, verbose=True, **kwargs):
    q, k, v, dO = get_attention_inputs(b, h, n, dv, dt, pad_multiple=192)
    
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y, l_vec = tk.mha_forward(q, k, v, causal)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
        assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None

    if is_forwards:
        return y, tot

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        q_grad, k_grad, v_grad = tk.mha_backward(q, k, v, y, l_vec, dO, causal)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
        assert not np.isnan(q_grad.float().cpu()).any(), "NaN values detected in output 'q_grad'"
        assert not np.isinf(q_grad.float().cpu()).any(), "Inf values detected in output 'q_grad'"
        assert not np.isnan(k_grad.float().cpu()).any(), "NaN values detected in output 'k_grad'"
        assert not np.isinf(k_grad.float().cpu()).any(), "Inf values detected in output 'k_grad'"
        assert not np.isnan(v_grad.float().cpu()).any(), "NaN values detected in output 'v_grad'"
        assert not np.isinf(v_grad.float().cpu()).any(), "Inf values detected in output 'v_grad'"

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        q_grad = None
        k_grad = None
        v_grad = None

    return (q_grad, k_grad, v_grad), tot


def fa3_test(dt, b, h, n, dv, causal, is_forwards, verbose=True, **kwargs):
    q, k, v, dO = get_attention_inputs(b, h, n, dv, dt)
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    dO = dO.transpose(1,2)
    
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y, _ = flash_attn_func3(q, k, v, causal=causal)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
        assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        tot = -1
        y = None

    if is_forwards:
        return y, tot

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y.backward(dO)

        q_grad = q.grad
        k_grad = k.grad
        v_grad = v.grad

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except:
        if verbose:
            print(f"Error: {sys.exc_info()[0]}")
        tot = -1
        q_grad = None
        k_grad = None
        v_grad = None

    return (q_grad, k_grad, v_grad), tot



from functools import partial
IMPLEMENTATIONS = {
    "fwd_PT_c=t": partial(pytorch_test, causal=True, is_forwards=True),
    "fwd_FA2_c=t": partial(fa2_test, causal=True, is_forwards=True),
    "fwd_FA3_c=t": partial(fa3_test, causal=True, is_forwards=True),
    "fwd_TK_c=t": partial(tk_test, causal=True, is_forwards=True),
    "fwd_PT_c=f": partial(pytorch_test, causal=False, is_forwards=True),
    "fwd_FA2_c=f": partial(fa2_test, causal=False, is_forwards=True),
    "fwd_FA3_c=f": partial(fa3_test, causal=False, is_forwards=True),
    "fwd_TK_c=f": partial(tk_test, causal=False, is_forwards=True),
}

IMPLEMENTATIONS_BWD = {
    "bwd_PT_c=t": partial(pytorch_test, causal=True, is_forwards=False),
    "bwd_FA2_c=t": partial(fa2_test, causal=True, is_forwards=False),
    "bwd_FA3_c=t": partial(fa3_test, causal=True, is_forwards=False),
    "bwd_TK_c=t": partial(tk_test, causal=True, is_forwards=False),
    "bwd_PT_c=f": partial(pytorch_test, causal=False, is_forwards=False),
    "bwd_FA2_c=f": partial(fa2_test, causal=False, is_forwards=False),
    "bwd_FA3_c=f": partial(fa3_test, causal=False, is_forwards=False),
    "bwd_TK_c=f": partial(tk_test, causal=False, is_forwards=False),
}


