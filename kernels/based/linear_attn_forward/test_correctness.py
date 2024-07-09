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

loaded_correctly = []

# To import the ThunderKittens based kernels
try:
    sys.path.append("../linear_attn_forward/H100")
    import lin_attn as mod
    import lin_attn_4090 as mod_4090
    print(f"Successfully imported TK based_H100 kernel")
except:
    loaded_correctly.append('Based TK')
    mod = None
    print("Could not import TK based_H100 kernel; Uncomment it in methods dict below if you don't want to evaluate it.")

# To import the ThunderKittens JRT kernels
try:
    sys.path.append("H100")
    import prefill as mod_jrt
    print(f"Successfully imported TK JRT_H100 kernel")
except:
    loaded_correctly.append('JRT TK')
    mod_jrt = None
    print("Could not import TK JRT_H100 kernel; Uncomment it in methods dict below if you don't want to evaluate it.")


# To compare to Flash Linear Attention 
try:
    # Install from https://github.com/sustcsonglin/flash-linear-attention
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    print(f"Successfully imported flash_linear_attention")
except:
    loaded_correctly.append('FLA')
    fused_chunk_based, parallel_based, naive_parallel_based = None, None, None
    chunk_gla, fused_chunk_gla, fused_recurrent_gla = None, None, None
    print("Could not import flash_linear_attention; Uncomment it in methods dict below if you don't want to evaluate it.")

assert not loaded_correctly, f"Please install the necessary packages to run the benchmarks or comment out the benchmarks that you don't want.  Missing: {loaded_correctly}"


################ Based Versions ################

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, **kwargs: any):
        super().__init__()
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)
        self.rrd = math.sqrt(self.rd)
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor, chosen_terms = [0, 1, 2], add_scale=True):
        terms_list = []
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.to(torch.float32).unsqueeze(-1) * x.to(torch.float32).unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        if add_scale:
            x2 = x2 / (self.rd)
        terms = [ x[..., :1] ** 0 ]
        if add_scale:
            terms.append(x / self.rrd)
        else:
            terms.append(x)
        terms.append(x2)
        for i, term in enumerate(terms):
            if i in chosen_terms: terms_list.append(term)
        return torch.cat(terms_list, dim=self.head_dim_idx)


def pytorch_test_v1(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    # SA: note the torch.float32 conversions are very important 

    b, h, n, D = Q.shape
    print(f"{b=}, {h=}, {n=}, {D=}")
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))
    O2  = make_causal(O.to(torch.float32)**2)
    O1 = make_causal(O)

    T2  = torch.einsum("bhnm,bhmd->bhnd", O2.to(torch.float32), V.to(torch.float32))
    T1 = torch.einsum("bhnm,bhmd->bhnd", O1.to(torch.float32), V.to(torch.float32))  
    T0  = V.to(torch.float32).cumsum(dim=2)

    y  = T0 
    if add_scale:
        y += T1/math.sqrt(D) + T2/(2*D) 
    else:
        y += T1 + T2/2

    # KV states by term (a2)
    A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K.to(torch.float32),V.to(torch.float32),K.to(torch.float32)).cumsum(dim=2) / (math.sqrt(2)) 
    if add_scale:
        A2 = A2 / math.sqrt(D)
    A2 = A2[:, :, -1]
    A2 = rearrange(A2, 'b h e f d -> b h (e f) d')

    # KV states by term (a1)
    A1 = torch.einsum("bhnd,bhne->bhnde",K.to(torch.float32),V.to(torch.float32)).cumsum(dim=2) 
    if add_scale:
        A1 = A1 / math.sqrt(math.sqrt(D))
    A1 = A1[:, :, -1].transpose(2, 3)

    # KV states by term (a0)
    A0 = V.to(torch.float32).cumsum(dim=2)[:, :, -1]
    return y, A2, A1, A0


def pytorch_test_v2(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D)
    Q = feature_map(Q.to(torch.float32), add_scale=add_scale)
    K = feature_map(K.to(torch.float32), add_scale=add_scale)
    A_qk = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)) 
    A_qk = torch.tril(A_qk.to(torch.float32))
    y = torch.einsum("bhnm,bhme->bhne", A_qk.to(torch.float32), V.to(torch.float32))

    if add_norm:
        k_state = k.cumsum(dim=2)
        k_state = (q * k_state).sum(dim=-1) + 1e-12
        y = y / k_state

    return y


def pytorch_test_v3(dt, Q, K, V, d, verbose=True, add_norm=False,  add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D)

    # for the output
    q, k = feature_map(Q.to(torch.float32), add_scale=add_scale), feature_map(K.to(torch.float32), add_scale=add_scale)
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)

    # for the term 2 kv state (a2)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[2], add_scale=add_scale), feature_map(K.to(torch.float32), chosen_terms=[2], add_scale=add_scale)
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A2 = kv_state[:, :, -1].transpose(2, 3)

    # for the term 1 kv state (a1)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[1], add_scale=add_scale), feature_map(K.to(torch.float32), chosen_terms=[1], add_scale=add_scale)
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A1 = kv_state[:, :, -1]

    # for the term 0 kv state (a0)
    kv_state = (1 * v.to(torch.float32)).cumsum(dim=2)
    A0 = kv_state[:, :, -1].squeeze(-1)
    return out, A2, A1, A0


def fla_parallel_based_test(dt, q, k, v, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    y = parallel_based(q, k, v, add_scale, add_norm)
    return y


def based_kernel_test(dt, Q, K, V, d, verbose=True, add_scale=False, add_norm=False, output_state=False):
    b, h, n, d = Q.shape
    dv = V.shape[-1]
    o   = torch.zeros_like(V)
    kv_state_a2 = torch.zeros((b, h, dv, d*d), dtype=dt, device='cuda')
    kv_state_a1 = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    kv_state_a0 = torch.zeros((b, h, dv), dtype=dt, device='cuda')

    mod.based_fwd_tk(int(add_scale),int(output_state),Q,K,V,o,kv_state_a2,kv_state_a1,kv_state_a0)

    o += torch.zeros_like(o) # trigger an error if one exists
    kv_state_a2 = kv_state_a2.transpose(2,3)
    return o, kv_state_a2, kv_state_a1, kv_state_a0


################### Benchmarking and Correctness Tests ####################

def linear_attn_correct(dt):
    b = 4
    n = 2048
    h = 16
    d = 16
    dv = 64
    add_scale=True 
    add_norm=False
    output_state=True
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    pytorch_v1, kv_a2_v1, kv_a1_v1, kv_a0_v1  = pytorch_test_v1(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    # pytorch 2 uses a quadratic view so doesn't expose the recurrent state
    pytorch_v2                                = pytorch_test_v2(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    pytorch_v3, kv_a2_v3, kv_a1_v3, kv_a0_v3  = pytorch_test_v3(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    tk_outputs, kv_a2_tk, kv_a1_tk, kv_a0_tk  = based_kernel_test(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale, output_state=output_state)
    fla_parallel_out = fla_parallel_based_test(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)

    print(f"Note we find numerical differences upon inspecting the tensor outputs:\n")
    print(f"Checking outputs:")
    __eq("PyTorch v1 - PyTorch v2", pytorch_v1, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v2", pytorch_v3, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v1", pytorch_v3, pytorch_v1, debug=False)
    print()
    __eq("PyTorch v2 - Based TK", pytorch_v2, tk_outputs)
    __eq("PyTorch v1 - Based TK", pytorch_v1, tk_outputs, debug=False)
    __eq("PyTorch v2[0,0,:15] - Based TK[0,0,:15]", pytorch_v2[0,0,:105], tk_outputs[0,0,:105])
    __eq("PyTorch v1[0,0,:15] - Based TK[0,0,:15]", pytorch_v1[0,0,:105], tk_outputs[0,0,:105], debug=False)
    print("position 015:",pytorch_v2[0,0,15,:4])
    print("position 015",tk_outputs[0,0,15,:4])
    print("position 065:",pytorch_v2[0,0,65,:4])
    print("position 065:",tk_outputs[0,0,65,:4])
    print("position 105:",pytorch_v2[0,0,105,:4])
    print("position 105:",tk_outputs[0,0,105,:4])
    print()
    __eq("PyTorch v1 - FLA", pytorch_v1, fla_parallel_out, debug=False)
    __eq("PyTorch v2 - FLA", pytorch_v2, fla_parallel_out, debug=False)

    if output_state:
        print("\nChecking KV States (A2)")
        print(kv_a2_v3[0,0,0,:8])
        print(kv_a2_tk[0,0,0,:8])
        __eq("PyTorch v1 A2 - PyTorch v3 A2", kv_a2_v1, kv_a2_v3, debug=False)
        __eq("PyTorch v1 A2 - Based TK A2", kv_a2_v1, kv_a2_tk, debug=False)

        print("\nChecking KV States (A1)")
        print(f"{kv_a1_v1.shape=}, {kv_a1_v3.shape=}, {kv_a1_tk.shape=}")
        print(kv_a1_v3[0,1,15:17])
        print(kv_a1_tk[0,1,15:17])
        __eq("PyTorch v1 A1 - PyTorch v3 A1", kv_a1_v1, kv_a1_v3, debug=False)
        __eq("PyTorch v1 A1 - Based TK A1", kv_a1_v1, kv_a1_tk, debug=False)
        __eq("PyTorch v1 A1 - Based TK A1", kv_a1_v1[:,0], kv_a1_tk[:,0], debug=False)  # only one head gets written to

        print(f"\n Checking KV States (A0)")
        print(f"{kv_a0_v1.shape=}, {kv_a0_v3.shape=}, {kv_a0_tk.shape=}")
        __eq("PyTorch v1 A0 - PyTorch v3 A0", kv_a0_v1, kv_a0_v3, debug=False)
        __eq("PyTorch v1 A0 - PyTorch v3 A0", kv_a0_v1, kv_a0_tk, debug=False)

        # combining taylor expansion terms
        kv_state = torch.concat((kv_a2_tk, kv_a1_tk.transpose(2,3), kv_a0_tk.unsqueeze(2)), dim=-2)

if __name__ == "__main__":
    methods = {
            'Based PyTorch': pytorch_test_v2,
            'Based CUDA': based_kernel_test,
            'Based Triton': fla_parallel_based_test,
        }
    print("Correctness test...")
    linear_attn_correct(torch.bfloat16)

