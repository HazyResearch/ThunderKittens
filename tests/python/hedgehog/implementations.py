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


################ Hedgehog ################

D_QK = 128
D_VO = 128

def get_flops(batch, seqlen, headdim, nheads):
    # https://docs.google.com/spreadsheets/d/1sKBZBHMX_ABfu4XNfGgHM8xD67mtnJY_9lAaTGHtKC0/edit?gid=0#gid=0

    featuredim = 64
    headdim = 128

    def get_masks(window_size: int, q_len: int, k_len: int, device ='cpu') -> tuple[torch.Tensor]:
        """
        Return masks for softmax and linear attention terms
        -> 1 is include, 0 is ignore
        """
        import math
        l = window_size
        m = math.ceil(max(q_len, k_len) / window_size)
        mask = torch.block_diag(*[torch.ones((l, l), )] * m)
        mask += torch.roll(mask, -l, -1) # this adds the terracing
        if mask.shape[0] > q_len:
            mask = mask[-q_len:]
        if mask.shape[1] > k_len:
            mask = mask[:, -k_len:]
        mask_swa, mask_lin = torch.tril(mask), torch.tril(1 - mask)

        num_mask_swa = mask_swa.sum().item()
        num_mask_lins = mask_lin.sum().item()
        return num_mask_swa, num_mask_lins
    
    # featurization 
    expanded_dim = featuredim * 2
    lin_flops = 2 * batch * seqlen * nheads * featuredim * headdim # compute feature map on q and k
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # apply a nonlinearity
    # numerator for linear
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (k * v) pointwise
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (cumsum) adds
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum) pointwise
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1) sum 
    # denominator for linear 
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (k) cumsum 
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (q * k.cumsum) pointwise

    # attention 
    attn_flops = 4 * batch * seqlen**2 * nheads * headdim // 2 # 2 is for causality
    
    # take proportions
    terraced_window_size = 64
    num_mask_swa, num_mask_lin = get_masks(terraced_window_size, seqlen, seqlen)
    pct_swa = num_mask_swa / (num_mask_swa + num_mask_lin)
    pct_lin = num_mask_lin / (num_mask_swa + num_mask_lin)

    f = (attn_flops * pct_swa) + (lin_flops * pct_lin)
    return f


def generate_inputs(testname, B, H, N):
    alphas = torch.rand((H,), dtype=torch.float32, device='cuda')*3
    betas = torch.rand((H,), dtype=torch.float32, device='cuda')
    if testname == 'randn':
        q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
        k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
        v = (torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
        qmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
        kmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
    elif testname == 'ones':
        q = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'qk_test': # this test breaks if sequence length is set <128
        q = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
        q = q.to(dtype=torch.bfloat16, device='cuda')
        k = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
        k = k.to(dtype=torch.bfloat16, device='cuda')
        v = torch.eye(D_VO).reshape((1,1,D_VO,D_VO)).repeat(B, H, N//D_VO, 1)*10
        v = v.to(dtype=torch.bfloat16, device='cuda')
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'v_or':
        q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.arange(D_VO*2, dtype=torch.bfloat16, device='cuda')/D_VO).reshape((1,1,2,-1)).repeat(B, H, N//2, 1) / 5
        qmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
        kmap = (torch.ones((H, D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    elif testname == 'dbg':
        q = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        k = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
        v = (torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 10
        qmap = (torch.eye(64, dtype=torch.bfloat16, device='cuda')).repeat((H,2,1))
        kmap = (torch.eye(64, dtype=torch.bfloat16, device='cuda')).repeat((H,2,1))
    else:
        print('bad testname')
        sys.exit(0)
    return q, k, v, qmap, kmap, alphas, betas


def hedgehog_pytorch_test(dt, b, h, n, dv, verbose=True, **kwargs):
    Q, K, V, Qmap, Kmap, alphas, betas = generate_inputs('randn', b, h, n)
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

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    
    try:
        torch.cuda.synchronize()
        start_events[0].record()

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
        kv_state = torch.einsum('bhlf,bhld->bhfd', Ks[:,:,:-64,:], V[:,:,:-64,:]).to(torch.float32).detach()
        k_state  = Ks[:,:,:-64,:].to(torch.float32).sum(dim=-2).detach()

        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]

    except:
        tot = -1
        out = None
        kv_state = None
        k_state = None
    # state_size = kv_state.numel()
    # print(f"state_size: {state_size}")
    
    return (out, kv_state, k_state), tot


def profile_tk_hedgehog(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v, qmap, kmap, alphas, betas =  generate_inputs('randn', b, h, n)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    try:
        torch.cuda.synchronize()
        start_events[0].record()

        o, kv_state, k_state = tk.hedgehog(q, k, v, qmap, kmap, alphas, betas)
        
        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
        assert not np.isnan(o.float().cpu()).any(), "NaN values detected in output 'o'"
        assert not np.isinf(o.float().cpu()).any(), "Inf values detected in output 'o'"
    except Exception as e:
        print(e)
        tot = -1
        o = None

    return (o, kv_state, k_state), tot


def profile_hedgehog_fla(dt, b, h, n, dv, verbose=True, **kwargs):
    q, k, v, qmap, kmap, alphas, betas =  generate_inputs('randn', b, h, n)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

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
    # print("state_size", state_size)

    torch.cuda.synchronize()
    start_events[0].record()

    q = pytorch_softmax_gt(q, qmap)
    k = pytorch_softmax_gt(k, kmap)
    o, _ = fused_chunk_linear_attn(q, k, v, scale=False, normalize=False)

    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    return o, tot


IMPLEMENTATIONS = {
    "torch_hedgehog": hedgehog_pytorch_test,
    "fla_triton_hedgehog": profile_hedgehog_fla,
    "tk_hedgehog": profile_tk_hedgehog,
}

