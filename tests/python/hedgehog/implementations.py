import os
import numpy as np
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

try:
    # To compare to Faster Transformers
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported causal_attention_cuda")
except:
    causal_dot_product = None
    print("Could not import causal_attention_cuda")

try:
    from fla.ops.linear_attn import fused_chunk_linear_attn
    print(f"Successfully imported flash_linear_attention")
except:
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
    q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
    k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
    v = (torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
    qmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
    kmap = (torch.randn((H, D_QK, 64), dtype=torch.bfloat16, device='cuda'))
    return q, k, v, qmap, kmap, alphas, betas


def hedgehog_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):

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
    
    def pytorch_method(Q, K, V, Qmap, Kmap, alphas, betas):
        Qs = pytorch_softmax_gt(Q, Qmap)
        Ks = pytorch_softmax_gt(K, Kmap)
        a_lin = torch.einsum('bhmd,bhnd->bhmn', Qs, Ks).to(torch.float32)
        a_exp = torch.einsum('bhmd,bhnd->bhmn', Q, K).to(torch.float32)
        a_lin *= lin_mask[:,:,:a_lin.shape[2], :a_lin.shape[3]] * alphas.reshape((1,-1,1,1)) # mask
        a_exp += exp_mask[:,:,:a_exp.shape[2], :a_exp.shape[3]] # subtract infinity
        a_exp -= a_exp.amax(dim=-1, keepdim=True)
        a_exp = torch.exp(a_exp / (128**.5)) * betas.reshape((1,-1,1,1))

        a = a_exp + a_lin
        a = (a / (a.sum(dim=-1, keepdim=True)+1e-6)).to(torch.bfloat16) # normalize
        
        o = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
        kv_state = torch.einsum('bhlf,bhld->bhfd', Ks[:,:,:-64,:], V[:,:,:-64,:]).to(torch.float32).detach()
        k_state  = Ks[:,:,:-64,:].to(torch.float32).sum(dim=-2).detach()
        
        return o, k_state, kv_state
    
    if torch_compile and method_str == "pytorch":
        pytorch_method = torch.compile(pytorch_method)

    for stage in ['warmup', 'timed']:

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
        for i in range(num_iters):
            Q, K, V, Qmap, Kmap, alphas, betas = generate_inputs('randn', b, h, n)
            generator_mat = torch.block_diag(*[torch.ones((64,64), device='cuda')]*256) # 16384 x 16384 should be big enough
            generator_mat += torch.roll(generator_mat, -64, -1) # this adds the terracing
            lin_mask = torch.tril(1-generator_mat).reshape((1,1,16384,16384))
            exp_mask = torch.tril(generator_mat).reshape((1,1,16384,16384))
            exp_mask = 10000*exp_mask - 10000 

            try:
                if method_str == 'pytorch':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    
                    o, k_state, kv_state = pytorch_method(Q, K, V, Qmap, Kmap, alphas, betas)
                    
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'fla_triton_hedgehog':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    Q = pytorch_softmax_gt(Q, Qmap)
                    K = pytorch_softmax_gt(Q, Kmap)
                    o, _ = fused_chunk_linear_attn(Q, K, V, scale=False, normalize=False)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'tk_hedgehog':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    o, kv_state, k_state = tk.hedgehog(Q, K, V, Qmap, Kmap, alphas, betas)
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert False, f"Unknown method: {method_str}"

            except Exception as e:
                if verbose:
                    print(f"Error in {method_str}")    
                return None, -1

            torch.cuda.empty_cache()
            
    assert not np.isnan(o.float().cpu()).any(), "NaN values detected in output 'o'"
    assert not np.isinf(o.float().cpu()).any(), "Inf values detected in output 'o'"
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    return (o), tot


IMPLEMENTATIONS = {
    "pytorch": partial(hedgehog_test, causal=True, is_forwards=True, method_str="pytorch"),
    "fla_triton_hedgehog": partial(hedgehog_test, causal=True, is_forwards=True, method_str="fla_triton_hedgehog"),
    "tk_hedgehog": partial(hedgehog_test, causal=True, is_forwards=True, method_str="tk_hedgehog"),
}

NAME = "HEDGEHOG"
