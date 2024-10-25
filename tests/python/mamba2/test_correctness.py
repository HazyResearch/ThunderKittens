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


# Simple test
def test_correctness():
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 2, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = torch.randn(batch, seqlen, nheads, dtype=dtype, device=device).requires_grad_()
    dt_bias = torch.randn(nheads, dtype=dtype, device=device).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=dtype, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)
    z = None

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    # Pytorch
    D_factor = x * rearrange(D, "d -> d 1")
    _dt = F.softplus(dt + dt_bias)
    y_ref, _ = ssd_minimal_discrete(x*_dt.unsqueeze(-1), A*_dt, B, C, block_len=chunk_size)
    y_ref += D_factor

    # TK 
    _dt = F.softplus(dt + dt_bias)
    v = x*_dt.unsqueeze(-1)
    a = A*_dt
    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(v, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(a, "b h l -> b l h").to(torch.float32).contiguous()
    y_tk = tk.mamba2(q, k, v, a) # chunk size is 64
    y_tk = rearrange(y_tk, "b h l d -> b l h d").contiguous().to(dtype)
    y_tk += D_factor

    has_nan = torch.isnan(y_tk).any()
    has_inf = torch.isinf(y_tk).any()
    print(f"Has NaN: {has_nan}; Has Inf: {has_inf}")


    # set print options so I can see the difference
    torch.set_printoptions(sci_mode=False)

    breakpoint()

    # Check if the outputs are the same
    print(y[0,1000,4,:8])
    print(y_ref[0,1000,4,:8])
    print(y_tk[0,1000,4,:8])

    diff = torch.max(torch.abs(y - y_ref))
    close = torch.allclose(y, y_ref, atol=1e-4)
    print(f"PyTorch (FP32) - Triton (FP32): Max diff: {diff}; Close: {close}")

    diff = torch.max(torch.abs(y - y_tk))
    close = torch.allclose(y, y_tk, atol=1e-4)
    print(f"TK (BF16) - PyTorch (FP32): Max diff: {diff}; Close: {close}")

    diff = torch.max(torch.abs(y_ref - y_tk))
    close = torch.allclose(y_ref, y_tk, atol=1e-4)
    print(f"TK (BF16) -Triton (FP32): Max diff: {diff}; Close: {close}")

if __name__ == "__main__":
    test_correctness()

