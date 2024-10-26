import torch 
import torch.nn.functional as F
from einops import rearrange

from baselines.ssd_minimal import ssd_minimal_discrete 

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except ImportError:
    print(f"Failed to import mamba_chunk_scan_combined; Please 'pip install mamba_ssm'.")
    mamba_chunk_scan_combined = None


try:
    import thunderkittens as tk 
    print(f"Successfully imported thunderkittens")
except ImportError:
    print(f'Failed to import thunderkittens; Please "pip setup.py install".')
    tk = None


import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd(args...)` is deprecated..*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd(args...)` is deprecated*")




# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


# Simple test
def test_correctness():
    # torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    nheads = 16
    batch, seqlen, chunk_size,headdim = 16, 1024, 64,  64
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    # TK 
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)

    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(x*dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(A*dt, "b h l -> b l h").to(torch.float32).contiguous()


    for i in range(2000):
        print(f"{torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), torch.isnan(a).any()}")
        torch.cuda.synchronize()
        y_tk = tk.mamba2(q, k, v, a) # chunk size is 64
        torch.cuda.synchronize()
        y_tk = rearrange(y_tk, "b h l d -> b l h d").contiguous().to(dtype)
        has_nan = torch.isnan(y_tk).any()
        has_inf = torch.isinf(y_tk).any()
        num_nans = torch.sum(torch.isnan(y_tk))
        print(f"Has NaN: {has_nan}; Has Inf: {has_inf}; Num of NaNs: {num_nans} / {y_tk.numel()}")
        nan_positions = torch.nonzero(torch.isnan(y_tk))
        vals_gt_1e8 = torch.nonzero(y_tk > 1e8)
        if has_nan or has_inf or vals_gt_1e8.numel() > 0:
            breakpoint()

        x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
        A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
        B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
        C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
        D = torch.randn(nheads, dtype=dtype, device=device)
        
        q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        v = rearrange(x*dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        a = rearrange(A*dt, "b h l -> b l h").to(torch.float32).contiguous()

if __name__ == "__main__":
    test_correctness()

