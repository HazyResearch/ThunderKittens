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


# Simple test
def test_correctness():
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)

    # TK 
    q = C 
    k = B
    v = x*dt.unsqueeze(-1)
    a = A*dt
    q = rearrange(q, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(k, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(v, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(a, "b h l -> b l h").to(torch.float32).contiguous()
    y_tk = tk.mamba2(q, k, v, a) # chunk size is 64
    y_tk = rearrange(y_tk, "b h l d -> b l h d").to(torch.float32).contiguous()

    # Check if the outputs are the same
    print(y[0,0,0,:8])
    print(y_min[0,0,0,:8])
    print(y_tk[0,0,0,:8])

    diff = torch.max(torch.abs(y - y_min))
    close = torch.allclose(y, y_min, atol=1e-4)
    print(f"PyTorch-Triton: Max diff: {diff}; Close: {close}")

    diff = torch.max(torch.abs(y - y_tk))
    close = torch.allclose(y, y_tk, atol=1e-4)
    print(f"TK-PyTorch: Max diff: {diff}; Close: {close}")

    diff = torch.max(torch.abs(y_min - y_tk))
    close = torch.allclose(y_min, y_tk, atol=1e-4)
    print(f"TK-Triton: Max diff: {diff}; Close: {close}")

if __name__ == "__main__":
    test_correctness()


