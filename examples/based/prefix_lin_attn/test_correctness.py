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
sys.path.append('../../')
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
        
    def forward(self, x: torch.Tensor, chosen_terms = [0, 1, 2]):
        terms_list = []
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.to(torch.float32).unsqueeze(-1) * x.to(torch.float32).unsqueeze(-2)).flatten(start_dim=-2) / (self.r2 * self.rd)
        terms = [ x[..., :1] ** 0, x / self.rrd, x2 ]
        for i, term in enumerate(terms):
            if i in chosen_terms: terms_list.append(term)
        return torch.cat(terms_list, dim=self.head_dim_idx)


def pytorch_test_v1(dt, Q, K, V, d, verbose=True, **kwargs):
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

    y  = T0 + T1/math.sqrt(D) + T2/(2*D) 

    # KV states by term
    A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K.to(torch.float32),V.to(torch.float32),K.to(torch.float32)).cumsum(dim=2) / (math.sqrt(D) * math.sqrt(2))
    A2 = A2[:, :, -1]
    A2 = rearrange(A2, 'b h e f d -> b h (e f) d')
    return y, A2


def pytorch_test_v2(dt, Q, K, V, d, verbose=True, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D)
    Q, K = feature_map(Q.to(torch.float32)), feature_map(K.to(torch.float32))
    A_qk = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)) 
    A_qk = torch.tril(A_qk.to(torch.float32))
    y = torch.einsum("bhnm,bhme->bhne", A_qk.to(torch.float32), V.to(torch.float32))
    return y.to(dt)


def pytorch_test_v3(dt, Q, K, V, d, verbose=True, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D)

    # for the output
    q, k = feature_map(Q.to(torch.float32)), feature_map(K.to(torch.float32))
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)

    # for the term 2 kv stat (a2)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[2]), feature_map(K.to(torch.float32), chosen_terms=[2])
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)

    return out, last_kv_state


def fla_parallel_based_test(dt, q, k, v, d, verbose=True, **kwargs):
    y = parallel_based(q, k, v, False, False)
    return y


def based_kernel_test(dt, Q, K, V, d, verbose=True):
    b, h, n, d = Q.shape
    dv = V.shape[-1]
    o   = torch.zeros_like(V)
    kv_state = torch.zeros((b, h, dv, d*d), dtype=dt, device='cuda')
    mod.based_fwd_tk(Q,K,V,o,kv_state)
    o += torch.zeros_like(o) # trigger an error if one exists
    kv_state = kv_state.transpose(2,3)
    return o, kv_state


################### Benchmarking and Correctness Tests ####################

def linear_attn_correct(dt):
    b = 4
    n = 2048
    h = 16
    d = 16
    dv = 64
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.ones(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.ones(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')/dv

    pytorch_test_result, kv_state_v1 = pytorch_test_v1(dt, Q, K, V, d)
    pytorch_test_v2_result = pytorch_test_v2(dt, Q, K, V, d)
    pytorch_test_v3_result, kv_state_v3 = pytorch_test_v3(dt, Q, K, V, d) 
    based_kernel_test_result, based_kv_state = based_kernel_test(dt, Q, K, V, d)
    fla_parallel_based_test_result = fla_parallel_based_test(dt, Q, K, V, d)

    print(f"Note we find numerical differences upon inspecting the tensor outputs:")
    __eq("PyTorch v1 - PyTorch v2", pytorch_test_result, pytorch_test_v2_result, debug=False)
    __eq("PyTorch v3 - PyTorch v2", pytorch_test_v3_result, pytorch_test_v2_result, debug=False)
    __eq("PyTorch v3 - PyTorch v1", pytorch_test_v3_result, pytorch_test_result, debug=False)
    print()
    __eq("PyTorch v2 - Based TK", pytorch_test_v2_result, based_kernel_test_result, debug=False)
    __eq("PyTorch v1 - Based TK", pytorch_test_result, based_kernel_test_result, debug=False)
    print()
    __eq("PyTorch v1 - FLA", pytorch_test_result, fla_parallel_based_test_result, debug=False)
    __eq("PyTorch v2 - FLA", pytorch_test_v2_result, fla_parallel_based_test_result, debug=False)

    print("\nChecking KV States")
    print(kv_state_v1[0,0,0,:16])
    print(kv_state_v3[0,0,0,:16])
    print(based_kv_state[0,0,0,:16])
    __eq(
        "PyTorch v1 KV State - PyTorch v3 KV State", 
        kv_state_v1[0], 
        kv_state_v3[0], 
        debug=False
    )
    __eq(
        "PyTorch KV State - Based KV State", 
        kv_state_v1[0], 
        based_kv_state[0], 
        debug=False
    )
    # print(pytorch_test_result[0][0][0])
    # print(pytorch_test_v2_result[0][0][0])
    # for b in range(4):
    #     for h in range(16):
    #         for n in range(2048):
    #             res1 = pytorch_test_result[b][h][n]
    #             res2 = pytorch_test_v2_result[b][h][n]
    #             pytorch_diff = torch.abs(res1 - res2).max().item()
    #             if pytorch_diff > 1 and b > 0: breakpoint()
    # breakpoint()


if __name__ == "__main__":
    methods = {
            'Based PyTorch': pytorch_test_v2,
            'Based CUDA': based_kernel_test,
            'Based Triton': fla_parallel_based_test,
        }
    print("Correctness test...")
    linear_attn_correct(torch.bfloat16)

