import torch
import sys
import os
import time
import argparse

try:
    import thunderkittens as tk
except:
    print("Could not import thunderkittens")
    
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

###################  OUR UTILS ####################
# from plot import plot_results

from get_flops import (
    get_flops_based, get_flops_hedgehog, 
    get_flops_mamba2, 
    get_flops_attn, get_flops_fftconv,
    get_layernorm_flops,
    get_flops_rotary
)

################### KERNEL IMPORTS ####################

try:
    # To compare to Faster Transformers
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    causal_dot_product = None
    print("Could not import causal_attention_cuda")

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    from fla.ops.linear_attn import fused_chunk_linear_attn
    print(f"Successfully imported flash_linear_attention")
except:
    fused_chunk_based, parallel_based, naive_parallel_based = None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention. Install from https://github.com/sustcsonglin/flash-linear-attention")

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except:
    mamba_chunk_scan_combined = None 
    print("Could not import mamba_chunk_scan_combined. Please: pip install mamba_ssm")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print(f"Successfully imported selective_scan_fn")
except:
    selective_scan_fn = None 
    print("Could not import selective_scan_fn. Please: pip install mamba_ssm")

try:
    # To compare to Flash Attention
    from flash_attn import flash_attn_func
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention. Please: pip install flash-attn")

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

try:
    from baselines.layer_norm_triton import layer_norm_fn, RMSNorm
    print(f"Successfully imported layer_norm_fn")
except:
    layer_norm_fn = None
    print("Could not import layer_norm_fn. Please obtain from FlashAttention repo.")

try:
    from baselines.rotary import RotaryEmbedding
    print(f"Successfully imported RotaryEmbedding")
except:
    RotaryEmbedding = None
    print("Could not import RotaryEmbedding. Please obtain from FlashAttention repo.")


################ Flash Attention ################

def get_attention_inputs(b, h, n, dv, dt):
    q = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    k = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    v = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    return q, k, v


def flash_attention_test(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v = get_attention_inputs(b, h, n, dv, dt)
    q = torch.randn_like(v).transpose(1,2)
    k = torch.randn_like(v).transpose(1,2)
    v = torch.randn_like(v).transpose(1,2)

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
    # except Exception as e:
    #     tot = -1
    #     y = None
    return y, tot


################ Based ################

def get_based_inputs(b, h, n, dv, dt):
    feature_dim = 16
    Q   = torch.randn(b,h,n,feature_dim, dtype=dt, device='cuda')/feature_dim
    K   = torch.randn(b,h,n,feature_dim, dtype=dt, device='cuda')/feature_dim
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
    return Q, K, V


class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, **kwargs: any):
        super().__init__()
        import math
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)
        self.rrd = math.sqrt(self.rd)
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)


def pytorch_test(dt, b, h, n, dv, verbose=True, **kwargs):
    Q, K, v = get_based_inputs(b, h, n, dv, dt)
    d = 16

    feature_map = TaylorExp(input_dim=d)
    try:
        torch.cuda.synchronize()
        t0 = time.time()

        q, k = feature_map(Q), feature_map(K)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state = (k * v).cumsum(dim=2)  # causal 
        k_state = k.cumsum(dim=2)
        y = ((q * kv_state).sum(dim=-1)) 

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        tot = -1
        y= None
    return y, tot


def fast_transformer_test(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v = get_based_inputs(b, h, n, dv, dt)

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
        tot = -1
        y = None
    return y, tot


def fla_parallel_based_test(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v = get_based_inputs(b, h, n, dv, dt)

    try:
        torch.cuda.synchronize()
        t0 = time.time()

        y = parallel_based(q, k, v, False, False)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        tot = -1
        y = None
    return y, tot


def based_kernel_test(dt, b, h, n, dv, verbose=True, device='4090'):
    Q, K, V = get_based_inputs(b, h, n, dv, dt)

    o   = torch.zeros_like(V)
    b, h, n, d = Q.shape 
    dv = V.shape[-1]

    kv_state_a2 = torch.empty((b, h, d*d, dv), dtype=torch.bfloat16, device='cuda')
    kv_state_a1 = torch.empty((b, h, dv, d), dtype=torch.bfloat16, device='cuda')
    kv_state_a0 = torch.empty((b, h, d), dtype=torch.bfloat16, device='cuda')

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        o = tk.based( Q, K, V )
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        tot = -1
        o = None

    return o, tot


################ Mamba ################

def get_mamba2_inputs(dt, b, h, n, dv):
    # Set so Mamba-2's (2 * dim * dstate) is ~ Based's (dim * (1 + 16 + 16^2))
    d_state = 128
    dv_mamba = dv * 2   # we double the hidden size in each layer

    delta = F.softplus(torch.randn(b, n, h, dtype=torch.float32, device='cuda') - 4)
    A = (-torch.exp(torch.rand(h, dtype=torch.float32, device='cuda')))
    k = torch.randn(b, n, 1, d_state, dtype=dt, device='cuda')
    q = torch.randn(b, n, 1, d_state, dtype=dt, device='cuda')
    v = torch.randn(b, n, h, dv_mamba, dtype=dt, device='cuda')
    return delta, A, k, q, v


def mamba2_kernel_test(dt, b, h, n, dv, verbose=True, **kwargs):
    d_state = 128
    chunk_size = 64
    delta, A, k, q, v = get_mamba2_inputs(dt, b, h, n, dv)
    
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        y = mamba_chunk_scan_combined(
            v, dt, A, k, q, 
            chunk_size=chunk_size, D=None
        )
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        tot = -1
        y = None
    return y, tot


def mamba1_kernel_test(dt, q, k, v, d, verbose=True, **kwargs):
    b, h, n, d = q.shape
    dv = v.shape[-1]
    dstate = 16 # from Mamba paper, note state is 8x smaller then Based
    try:
        dmodel = dv*h*2
        A = torch.randn(dmodel, dstate, dtype=torch.float32, device=q.device)
        x = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=q.device)
        dt = torch.randn(b, dmodel,n, dtype=torch.bfloat16, device=q.device)    
        B = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=q.device)
        C = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=q.device)
        D = torch.randn(dmodel, dtype=torch.bfloat16, device=q.device)
        z = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=q.device)
        dt_proj_bias = torch.randn(dmodel, dtype=torch.bfloat16, device=q.device)

        torch.cuda.synchronize()
        t0 = time.time()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except Exception as e:
        tot = -1
        y = None
    return y, tot


################ Hedgehog ################

def get_hedgehog_inputs(dt, b, h, n, dv):
    dtype = dt
    device = 'cuda'
    feature_dim = 64
    value_dim = 128 

    q = torch.randn(b, h, n, 2*feature_dim, dtype=dtype, device=device)
    k = torch.randn(b, h, n, 2*feature_dim, dtype=dtype, device=device)
    v = torch.randn(b, h, n, value_dim, dtype=dtype, device=device)
    o   = torch.zeros_like(v)

    kv_state = torch.empty((b, h, 2*feature_dim, value_dim), dtype=torch.float32, device='cuda')
    k_state = torch.empty((b, h, 2*feature_dim), dtype=torch.float32, device='cuda')
    alphas = torch.rand((h,), dtype=torch.float32, device='cuda')*3
    betas = torch.rand((h,), dtype=torch.float32, device='cuda')
    qmap = (torch.randn((h, 2*feature_dim, feature_dim), dtype=torch.bfloat16, device='cuda'))
    kmap = (torch.randn((h, 2*feature_dim, feature_dim), dtype=torch.bfloat16, device='cuda'))

    return q, k, v, o, k_state, kv_state, qmap, kmap, alphas, betas


def hedgehog_pytorch_test(dt, b, h, n, dv, verbose=True, **kwargs):
    Q, K, V, o, k_state, kv_state, Qmap, Kmap, alphas, betas = get_hedgehog_inputs(dt, b, h, n, dv)
    generator_mat = torch.block_diag(*[torch.ones((64,64), device='cuda')]*256) # 16384 x 16384 should be big enough
    generator_mat += torch.roll(generator_mat, -64, -1) # this adds the terracing
    lin_mask = torch.tril(1-generator_mat).reshape((1,1,16384,16384))
    exp_mask = torch.tril(generator_mat).reshape((1,1,16384,16384))
    exp_mask = 10000*exp_mask - 10000 

    def pytorch_softmax_gt(x, map_mat, label=None):

        x = torch.einsum('bhmd,hdn->bhmn', x, map_mat)

        x_pos = x
        x_neg = -x

        x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
        x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
        
        x_pos = x_pos - x_pos_max
        x_neg = x_neg - x_neg_max
        
        x_pos_num = torch.exp(x_pos)
        x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
        
        x_neg_num = torch.exp(x_neg)
        x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
        
        x_pos = x_pos_num / x_pos_den
        x_neg = x_neg_num / x_neg_den
        
        x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
        
        return x
    
    torch.cuda.synchronize()
    t0 = time.time()

    Qs = pytorch_softmax_gt(Q, Qmap)
    Ks = pytorch_softmax_gt(K, Kmap)
    
    a_lin = torch.einsum('bhmd,bhnd->bhmn', Qs, Ks).to(torch.float32)
    a_exp = torch.einsum('bhmd,bhnd->bhmn', Q, K).to(torch.float32)
    # mask
    a_lin *= lin_mask[:,:,:a_lin.shape[2], :a_lin.shape[3]] * alphas.reshape((1,-1,1,1)) # zero out unwanted entries
    a_exp += exp_mask[:,:,:a_exp.shape[2], :a_exp.shape[3]] # subtract infinity off of unwanted entries
    a_exp -= a_exp.amax(dim=-1, keepdim=True)
    a_exp = torch.exp(a_exp / (128**.5)) * betas.reshape((1,-1,1,1))

    a = a_exp + a_lin
    a = (a / (a.sum(dim=-1, keepdim=True)+1e-6)).to(torch.bfloat16) # normalize
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    kv_state = torch.einsum('bhlf,bhld->bhfd', Ks[:,:,:-64,:], V[:,:,:-64,:]).to(torch.float32).detach()
    k_state  = Ks[:,:,:-64,:].to(torch.float32).sum(dim=-2).detach()
    state_size = kv_state.numel()
    print(f"state_size: {state_size}")
    
    return out, tot


def profile_tk_hedgehog(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v, o, k_state, kv_state, qmap, kmap, alphas, betas = get_hedgehog_inputs(dt, b, h, n, dv)

    try:
        torch.cuda.synchronize()
        t0 = time.time()
        o, kv_state, k_state = tk.hedgehog(q, k, v, qmap.unsqueeze(0), kmap.unsqueeze(0), alphas, betas)
        torch.cuda.synchronize()
        t1 = time.time()
        o += torch.zeros_like(o) # trigger an error if one exists
        tot = t1-t0
        assert not np.isnan(o.float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(o.float().cpu()).any(), "Inf values detected in output 'o'"
    except Exception as e:
        print(e)
        tot = -1
        o = None

    return o, tot

def profile_hedgehog_fla(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v, o, k_state, kv_state, qmap, kmap, alphas, betas = get_hedgehog_inputs(dt, b, h, n, dv)

    def pytorch_softmax_gt(x, map_mat, label=None):

        x = torch.einsum('bhmd,hdn->bhmn', x, map_mat)

        x_pos = x
        x_neg = -x

        x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
        x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
        
        x_pos = x_pos - x_pos_max
        x_neg = x_neg - x_neg_max
        
        x_pos_num = torch.exp(x_pos)
        x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
        
        x_neg_num = torch.exp(x_neg)
        x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
        
        x_pos = x_pos_num / x_pos_den
        x_neg = x_neg_num / x_neg_den
        
        x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
        
        return x

    value_dim = dv
    expand_k_factor = 1
    key_dim = int(expand_k_factor * dv)
    state_size = b* key_dim * value_dim * h
    print("state_size", state_size)

    q = torch.randn(b, h, n, key_dim, dtype=dt, device='cuda')
    k = torch.randn(b, h, n, key_dim, dtype=dt, device='cuda')
    v = torch.randn(b, h, n, value_dim, dtype=dt, device='cuda')

    torch.cuda.synchronize()
    t0 = time.time()

    q = pytorch_softmax_gt(q, qmap)
    k = pytorch_softmax_gt(k, kmap)
    o, _ = fused_chunk_linear_attn(q, k, v, scale=False, normalize=False)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    return o, tot


################ FFT Convolution ################


def get_fft_inputs(dt, b, h, n, dv):
    d_model = (h * dv)
    u = torch.randn((b, d_model, n), dtype=dt).to('cuda')
    k = torch.randn((d_model, n), dtype=torch.float32).to('cuda')
    return u, k


def fftconv_tk_test(dt, b, h, n, dv, verbose=True, **kwargs):
    if n not in [1024, 4096]:
        print(f"Sequence length {n} not supported by TKFFTConv")
        return None, -1
    
    u, k = get_fft_inputs(dt, b, h, n, dv)
    tk_fft = TKFFTConv(k, seqlen=n, H=(h*dv), dtype=u.dtype).to(u.device)

    torch.cuda.synchronize()
    t0 = time.time()

    y_tk = tk_fft(u, k)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0

    y_ref = ref_fftconv(u, k, n)
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



################## Layer norm ######################


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


#################### Rotary ########################

def get_rotary_inputs(b, h, n, dv, dt):
    d_model = h * dv
    head_dim = 128
    rotary_emb_fraction = 1
    rotary_emb_dim = head_dim * rotary_emb_fraction
    rotary_emb_base = 10000
    rotary_max_seqlen = None
    rotary_emb_scale_base=None
    rotary_emb_interleaved=False

    assert RotaryEmbedding is not None, "RotaryEmbedding not imported"

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device='cuda',
    )

    h = d_model // head_dim
    qkv = torch.randn((b, n, 3, h, head_dim), dtype=dt, device='cuda')
    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached
    return qkv, cos, sin, rotary_max_seqlen, rotary_emb

from baselines.rotary import apply_rotary_emb
def apply_flash_rotary(dt, b, h, n, dv):
    # Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L957

    qkv, cos, sin, rotary_max_seqlen, rotary_emb  = get_rotary_inputs(b, h, n, dv, dt)
    q = qkv[:,:,0]
    k = qkv[:,:,1]

    torch.cuda.synchronize()
    start = time.time()

    qkv_emb_flash = apply_rotary_emb(
        q, cos, sin
    )

    qkv_emb_flash = apply_rotary_emb(
        k, cos, sin
    )

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start
    return qkv_emb_flash, tot


def apply_rotary_emb_torch(dt, b, h, n, dv):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """

    qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)

    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    o_dt = qkv.dtype
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= qkv.shape[-1]
    interleaved = False
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]

    torch.cuda.synchronize()
    start = time.time()

    # for q
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    out =  torch.cat(
        [q[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(q[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), q[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)

    # for k 
    out =  torch.cat(
        [k[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(k[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), k[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start 
    return out, tot


def apply_rotary_emb_tk(dt, b, h, n, dv):
    qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)
    q = qkv[:,:,0].transpose(1,2).clone()   # torch.Size([b, n, 3, h, dv])
    k = qkv[:,:,1].transpose(1,2).clone()
    out = torch.zeros_like(q)

    cos = cos.contiguous()
    sin = sin.contiguous()
    q = q.contiguous()
    k = k.contiguous()

    torch.cuda.synchronize()
    start = time.time()

    out = tk.fused_rotary(q, cos, sin)
    out = tk.fused_rotary(k, cos, sin)

    torch.cuda.synchronize()
    end = time.time()
    tot = end-start
    return out, tot


############## Efficiency Measurements #############


#  Function to calculate the efficiency in teraflops
def efficiency(flops, time):
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms


def measure_efficiency(dt, n, method_name, method, verbose=False, use_ones=False, profile=False):
    num_iters = 10
    method2timing = defaultdict(dict)

    b = 16
    h = 16
    dv = 64
    print(f"{b=}, {n=}, {h=}, {dv=}")

    if 'based' in method_name:
        flops = get_flops_based(b, n, dv, h)
    elif 'mamba2' in method_name:
        flops = get_flops_mamba2(b, n, dv, h)
    elif 'hedgehog' in method_name:
        flops = get_flops_hedgehog(b, n, dv, h)
    elif 'flash att' in method_name:
        flops = get_flops_attn(b, n, dv, h)
    elif 'fftconv' in method_name:
        flops = get_flops_fftconv(b, n, dv, h)
    elif 'layernorm' in method_name:
        flops = get_layernorm_flops(b, n, dv, h)
    elif 'rotary' in method_name:
        flops = get_flops_rotary(b, n, dv, h)

    try:
        lst = [method(dt, b, h, n, dv) for _ in range(num_iters)]
        lst_time = [x[-1] for x in lst][2:] # skip the first two iterations (warmup)
        _time = median(lst_time)
    except:
        _time = -1

    microseconds = _time * 1000000
    eff = efficiency(flops, microseconds)
    print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, Time: {_time:.4f} s and FLOPS: {flops:.2f}")
    torch.cuda.empty_cache()
    return eff, _time

if __name__ == "__main__":
    print("Benchmarking the kernels...")
    methods = {
        # based
        # 'based_torch': pytorch_test,
        # 'based_tk': based_kernel_test,                    # In TK-1
        # 'based_fla': fla_parallel_based_test,
            
        # attention 
        # 'flash attention 2': flash_attention_test,

        # mamba-2
        # "mamba1": mamba1_kernel_test, # not implemented
        # "mamba2": mamba2_kernel_test,                     # In TK-2

        # hedgehog
        # "hedgehog_fla": profile_hedgehog_fla,
        "hedgehog_tk": profile_tk_hedgehog,
        # "hedgehog_pytorch": hedgehog_pytorch_test,        # In TK-1
        
        # fft convolution 
        # 'fftconv_pytorch': fftconv_pytorch_test,
        # 'fftconv_cutlass': fftconv_cutlass_test,
        # 'fftconv_tk': fftconv_tk_test,                    # In TK-2

        # layernorm 
        # 'layernorm torch': pytorch_layernorm_test,
        # 'layernorm triton': triton_layer_norm_test,
        # 'layernorm tk': layer_norm_test,                  # In TK-1

        # rotary
        # 'rotary_flash': apply_flash_rotary,
        # 'rotary_torch': apply_rotary_emb_torch,
        # 'rotary_tk': apply_rotary_emb_tk                    # In TK-2
    }

    method2tflops = {}
    method2timing = {}
    for m, method in methods.items():
        flops_result = {}   
        timing_result = {}
        print(f"Method: {m}")
        for n in [
            1024, 
            # 2048, 
            4096, 
            # 8192, 
            # 16384
        ]:
            print(f"Sequence Length: {n}")
            tflops, timing = measure_efficiency(torch.bfloat16, n, m, method, verbose=False, use_ones=False, profile=False)
            if tflops > 0: 
                flops_result[n] = tflops
                timing_result[n] = timing
        method2tflops[m] = flops_result
        method2timing[m] = timing_result

        tag = m.split("_")[0]
        # print(f"Saving {tag} results")
        # import pickle 
        # with open(f"{tag}_method2tflops.pkl", "wb") as f:
        #     pickle.dump(method2tflops, f)
        # with open(f"{tag}_method2timing.pkl", "wb") as f:
        #     pickle.dump(method2timing, f)



    



