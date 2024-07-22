import torch
import sys
import os
import math
import time
import argparse
import pandas as pd
from collections import defaultdict
from statistics import median
import torch.nn as nn
from einops import rearrange

# These installs are for pulling in the TK source
sys.path.append('../../../')
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')


# To import the ThunderKittens based kernels
try:
    sys.path.append("../linear_attn_forward/H100")
    import lin_attn as mod
    import lin_attn_4090 as mod_4090
    print(f"Successfully imported TK based_H100 kernel")
except:
    mod = None
    print("Could not import TK based_H100 kernel.")

################ Based Versions ################

eps = 1e-12
TERMS = set([0, 1, 2])

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, add_scale=False, **kwargs: any):
        super().__init__()
        self.r2  = math.sqrt(2) if add_scale else 1 
        self.rd  = math.sqrt(input_dim) if add_scale else 1 
        self.rrd = math.sqrt(self.rd) if add_scale else 1 
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor, chosen_terms = [0, 1, 2]):
        terms_list = []
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.to(torch.float32).unsqueeze(-1) * x.to(torch.float32).unsqueeze(-2)).flatten(start_dim=-2) / (self.r2*self.rd)
        terms = [ x[..., :1] ** 0 ]
        terms.append(x / self.rrd)
        terms.append(x2)
        for i, term in enumerate(terms):
            if i in chosen_terms: terms_list.append(term)
        return torch.cat(terms_list, dim=self.head_dim_idx)


def pytorch_test_v1(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    # SA: note the torch.float32 conversions are very important 

    b, h, n, D = Q.shape
    rd = math.sqrt(D) if add_scale else 1 
    rrd = math.sqrt(rd) if add_scale else 1
    r2 = math.sqrt(2) if add_scale else 1

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
    if add_norm: 
        y = numerator.to(torch.float32) / ( denominator.to(torch.float32).unsqueeze(-1) + eps ).to(torch.bfloat16)
    else:
        y = numerator

    return y, A2, A1, A0, k_state_a2[:, :, -1], k_state_a1[:, :, -1], D0[:, :, -1]


def pytorch_test_v2(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D, add_scale=add_scale)
    Q = feature_map(Q.to(torch.float32))
    K = feature_map(K.to(torch.float32))
    A_qk = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)) 
    A_qk = torch.tril(A_qk.to(torch.float32))
    y = torch.einsum("bhnm,bhme->bhne", A_qk.to(torch.float32), V.to(torch.float32))

    k_state = K.to(torch.float32).cumsum(dim=2)
    if add_norm:
        den = (Q * k_state).sum(dim=-1) + eps
        y = y / den.unsqueeze(-1)

    return y, k_state[:,:,-1]


def pytorch_test_v3(dt, Q, K, V, d, verbose=True, add_norm=False,  add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D, add_scale=add_scale)

    # for the output
    q, k = feature_map(Q.to(torch.float32)), feature_map(K.to(torch.float32))
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)
    # overall k_state
    if add_norm:
        denom = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
        out = out / denom

    # for the term 2 kv state (a2)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[2]), feature_map(K.to(torch.float32), chosen_terms=[2])
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A2 = kv_state[:, :, -1].transpose(2, 3)
    D2 = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
    k_state_a2 = k.to(torch.float32).cumsum(dim=2)[:,:,-1].squeeze(-2)

    # for the term 1 kv state (a1)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[1]), feature_map(K.to(torch.float32), chosen_terms=[1])
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A1 = kv_state[:, :, -1]
    D1 = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
    k_state_a1 = k.to(torch.float32).cumsum(dim=2)[:,:,-1].squeeze(-2)

    # for the term 0 kv state (a0)
    kv_state = (1 * v.to(torch.float32)).cumsum(dim=2)
    A0 = kv_state[:, :, -1].squeeze(-1)
    K0 = torch.ones(Q[..., :1].to(torch.float32).shape).to(Q.device)
    D0 =  K0.cumsum(dim=2).squeeze(-1)[:,:,-1]

    return out, A2, A1, A0, k_state_a2, k_state_a1, D0


def pytorch_test_v4(dt, Q, K, V, d, verbose=True, add_norm=False,  add_scale=False, **kwargs):
    B, H, L, D = Q.shape

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))**2
    O2  = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2.to(torch.float32), V.to(torch.float32)).to(torch.float32)
    T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)))
    T1 = torch.einsum("bhnm,bhme->bhne", T1a.to(torch.float32), V.to(torch.float32))
    T0  = V.to(torch.float32).cumsum(dim=2)

    rd = math.sqrt(D) if add_scale else 1 
    rrd = math.sqrt(rd) if add_scale else 1
    r2 = math.sqrt(2) if add_scale else 1

    # Denominator
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q.to(torch.float32), Q.to(torch.float32)) / (rd * r2)
    K2 = torch.einsum("bhnd,bhne->bhnde", K.to(torch.float32), K.to(torch.float32)) / (rd * r2)
    k_state_a2 = K2.to(torch.float32).cumsum(dim=2)
    D2 = torch.einsum("bhnde,bhnde->bhn", Q2.to(torch.float32), k_state_a2.to(torch.float32)) 
    k_state_a1 =  K.to(torch.float32).cumsum(dim=2)
    D1 = torch.einsum("bhnd,bhnd->bhn", Q.to(torch.float32), k_state_a1.to(torch.float32))/ ((rrd) ** 2)

    K0 = torch.ones(Q[..., :1].to(torch.float32).shape).to(Q.device)
    D0 =  K0.to(torch.float32).cumsum(dim=2).squeeze(-1)
    
    num, den = [], []
    num.append(T0.to(torch.float32))
    num.append(T1.to(torch.float32) / (rrd * rrd))
    num.append(T2.to(torch.float32) / (rd * r2 * rd * r2))
    if add_norm: 
        den.append(D0.to(torch.float32).unsqueeze(-1))
        den.append(D1.to(torch.float32).unsqueeze(-1))
        den.append(D2.to(torch.float32).unsqueeze(-1))
    
    o = sum([n for i, n in enumerate(num) if i in TERMS])
    den = sum([n for i, n in enumerate(den) if i in TERMS])
    if add_norm: 
        o = o / (den + eps)

    k_state_a2 = rearrange(k_state_a2, 'b h n d e -> b h n (d e)')
    return o, k_state_a2[:,:,-1], k_state_a1[:,:,-1], D0[:,:,-1]


def based_kernel_test(dt, Q, K, V, d, verbose=True, add_scale=False, add_norm=False, output_state=False):
    b, h, n, d = Q.shape
    dv = V.shape[-1]
    o  = torch.zeros_like(V)

    kv_state_a2 = torch.zeros((b, h, d*d, dv), dtype=dt, device='cuda')
    kv_state_a1 = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    kv_state_a0 = torch.zeros((b, h, dv), dtype=dt, device='cuda')

    k_state_a2 = torch.zeros((b, h, d*d), dtype=dt, device='cuda')
    k_state_a1 = torch.zeros((b, h, d), dtype=dt, device='cuda')
    k_state_a0 = torch.ones((b, h), dtype=dt, device='cuda') * n

    mod.based_fwd_tk(
        int(0 in TERMS), int(1 in TERMS), int(2 in TERMS),
        int(add_scale),int(add_norm),int(output_state),
        Q,K,V,o,
        kv_state_a2,kv_state_a1,kv_state_a0,
        k_state_a2,k_state_a1
    )

    o += torch.zeros_like(o) # trigger an error if one exists
    return o, kv_state_a2, kv_state_a1, kv_state_a0, k_state_a2, k_state_a1, k_state_a0


################### Benchmarking and Correctness Tests ####################

def linear_attn_correct(dt):
    b = 1
    n = 2048
    h = 1
    head_idx = 0
    d = 16
    dv = 64
    add_scale=False     
    add_norm=True
    output_kv_state=False
    output_k_state=False
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.ones(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.ones(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')/dv

    # get outputs from different methods
    tk_outputs = None 
    fla_parallel_out = None
    pytorch_v1, kv_a2_v1, kv_a1_v1, kv_a0_v1, k_a2_v1, k_a1_v1, k_a0_v1  = pytorch_test_v1(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    # pytorch 2 uses a quadratic view so doesn't expose the recurrent state
    pytorch_v2, k_state_v2_full  = pytorch_test_v2(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    pytorch_v3, kv_a2_v3, kv_a1_v3, kv_a0_v3, k_a2_v3, k_a1_v3, k_a0_v3  = pytorch_test_v3(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    pytorch_v4, k_a2_v4, k_a1_v4, k_a0_v4  = pytorch_test_v4(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    tk_outputs, kv_a2_tk, kv_a1_tk, kv_a0_tk, k_a2_tk, k_a1_tk, k_a0_tk  = based_kernel_test(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale, output_state=output_kv_state)
    torch.set_printoptions(sci_mode=False)


    # check overall outputs
    print(f"Note we find numerical differences upon inspecting the tensor outputs:\n")
    print(f"Checking outputs:")
    __eq("PyTorch v1 - PyTorch v2", pytorch_v1, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v2", pytorch_v3, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v1", pytorch_v3, pytorch_v1, debug=False)
    __eq("PyTorch v4 - PyTorch v1", pytorch_v4, pytorch_v1, debug=False)
    __eq("PyTorch v4[:,:,:100] - PyTorch v1[:,:,:100]", pytorch_v4[:,:,:100], pytorch_v1[:,:,:100], debug=False)
    if tk_outputs is not None:
        print()
        # __eq("PyTorch v2 - Based TK", pytorch_v2, tk_outputs)
        __eq("PyTorch v1 - Based TK", pytorch_v1, tk_outputs, debug=False)
        __eq("PyTorch v4 - Based TK", pytorch_v4, tk_outputs, debug=False)
        __eq("PyTorch v1[0,0,:15] - Based TK[0,0,:15]", pytorch_v1[0,head_idx,:105], tk_outputs[0,head_idx,:105], debug=False)
        print(pytorch_v4[0,head_idx,0,:4])
        print(tk_outputs[0,head_idx,0,:4])
        print()

    print(pytorch_v4[0,head_idx,70:72,:4])
    print(tk_outputs[0,head_idx,70:72,:4])
    print()

    print(pytorch_v4[0,head_idx,128:130,:4])
    print(tk_outputs[0,head_idx,128:130,:4])
    print()

    print(pytorch_v4[0,head_idx,500:502,:4])
    print(tk_outputs[0,head_idx,500:502,:4])
    print()

    print(pytorch_v4[0,head_idx,1000:1004,:4])
    print(pytorch_v1[0,head_idx,1000:1004,:4])
    print()
    breakpoint()

    print("---"*10)

    if output_kv_state:
        print("\nChecking KV States (A2)")
        print(f"{kv_a2_v1.shape=}, {kv_a2_v3.shape=}")
        __eq("PyTorch v1 A2 - PyTorch v3 A2", kv_a2_v1, kv_a2_v3, debug=False)
        if tk_outputs is not None: 
            __eq("PyTorch v1 A2 - Based TK A2", kv_a2_v1, kv_a2_tk, debug=False)
            print(kv_a2_tk[0,head_idx,0,:8])
            print(kv_a2_v1[0,head_idx,0,:8])

        print("\nChecking KV States (A1)")
        print(f"{kv_a1_v1.shape=}, {kv_a1_v3.shape=}")
        __eq("PyTorch v1 A1 - PyTorch v3 A1", kv_a1_v1, kv_a1_v3, debug=False)
        if tk_outputs is not None:
            __eq("PyTorch v1 A1 - Based TK A1", kv_a1_v1, kv_a1_tk, debug=False)
            __eq("PyTorch v1 A1 - Based TK A1", kv_a1_v1[:,0], kv_a1_tk[:,0], debug=False)  
            print(kv_a1_tk[0,head_idx,0,:8])
            print(kv_a1_v1[0,head_idx,0,:8])

        print(f"\nChecking KV States (A0)")
        print(f"{kv_a0_v1.shape=}, {kv_a0_v3.shape=}")
        __eq("PyTorch v1 A0 - PyTorch v3 A0", kv_a0_v1, kv_a0_v3, debug=False)
        if tk_outputs is not None:
            __eq("PyTorch v1 A0 - PyTorch TK A0", kv_a0_v1, kv_a0_tk, debug=False)
            print(kv_a0_tk[0,head_idx,:8])
            print(kv_a0_v1[0,head_idx,:8])

        # combining taylor expansion terms
        if tk_outputs is not None: 
            kv_state = torch.concat((kv_a2_tk, kv_a1_tk.transpose(2,3), kv_a0_tk.unsqueeze(2)), dim=-2)

    print("---"*10)

    if output_k_state:
        print("\nChecking K States (D2)")
        print(f"{k_a2_v1.shape=}, {k_a2_v3.shape=}")
        __eq("PyTorch v1 D2 - PyTorch v3 D2", k_a2_v1, k_a2_v3, debug=False)
        __eq("PyTorch v1 D2 - PyTorch v4 D2", k_a2_v1, k_a2_v4, debug=False)
        if tk_outputs is not None: 
            __eq("PyTorch v1 D2 - PyTorch tk D2", k_a2_v1, k_a2_tk, debug=False) # SA: currently has sporadic nans
            __eq("PyTorch v1 D2 [0,0,:64] - PyTorch tk D2 [0,0,:64]", k_a2_v1[0,0,:64], k_a2_tk[0,0,:64], debug=False)
            __eq("PyTorch v1 D2 [0,0]     - PyTorch tk D2 [0,0]    ", k_a2_v1[0,0], k_a2_tk[0,0], debug=False)
            nan_count = torch.isnan(k_a2_tk).sum().item()
            inf_count = len(
                [i for i, a in enumerate(k_a2_tk.to(torch.float32).flatten().cpu().numpy()) if abs(a) > 1000000]
            )
            print(k_a2_tk[0,head_idx,:8])
            print(k_a2_v1[0,head_idx,:8])
            print(f"Number of nans: {nan_count}; number of infs: {inf_count}")
            if inf_count: breakpoint()

        print("\nChecking K States (D1)")
        print(f"{k_a1_v1.shape=}, {k_a1_v3.shape=}")
        __eq("PyTorch v1 D1 - PyTorch v3 D1", k_a1_v1, k_a1_v3, debug=False)
        __eq("PyTorch v1 D1 - PyTorch v4 D1", k_a1_v1, k_a1_v4, debug=False)
        if tk_outputs is not None:
            __eq("PyTorch v1 D1 - PyTorch TK D1", k_a1_v1, k_a1_tk, debug=False)
            __eq("PyTorch v3 D1 - PyTorch TK D1", k_a1_v3, k_a1_tk, debug=False)
            print(k_a1_tk[0,head_idx,:8])
            print(k_a1_v1[0,head_idx,:8])

        print(f"\nChecking K States (D0)")
        print(f"{k_a0_v1.shape=}")
        __eq("PyTorch v1 D0 - PyTorch v3 D0", k_a0_v1, k_a0_v3, debug=False)
        __eq("PyTorch v1 D0 - PyTorch v4 D0", k_a0_v1, k_a0_v4, debug=False)
        if tk_outputs is not None:
            __eq("PyTorch v1 D0 - PyTorch TK D0", k_a0_v1, k_a0_tk, debug=False)
            print(k_a0_tk[0,head_idx])
            print(k_a0_v1[0,head_idx])

        print(f"\nChecking K States (Full)")
        k_state_v1_full = torch.concat([k_a0_v1.unsqueeze(-1), k_a1_v1, k_a2_v1], dim=-1)
        k_state_v3_full = torch.concat([k_a0_v3.unsqueeze(-1), k_a1_v3, k_a2_v3], dim=-1)
        # k_state_v4_full = torch.concat([k_a0_v4.unsqueeze(-1), k_a1_v4, k_a2_v4], dim=-1)
        if tk_outputs is not None:
            k_state_tk_full = torch.concat([k_a0_tk.unsqueeze(-1), k_a1_tk, k_a2_tk], dim=-1)
        print(f"{k_state_v1_full.shape=}, {k_state_v2_full.shape=}")
        __eq("PyTorch v1 - PyTorch v2", k_state_v1_full, k_state_v2_full, debug=False)
        __eq("PyTorch v1 - PyTorch v3", k_state_v1_full, k_state_v3_full, debug=False)
        # __eq("PyTorch v1 - PyTorch v4", k_state_v1_full, k_state_v4_full, debug=False)
        if tk_outputs is not None:
            __eq("PyTorch v1 - PyTorch tk", k_state_v1_full, k_state_tk_full, debug=False)

if __name__ == "__main__":
    linear_attn_correct(torch.bfloat16)