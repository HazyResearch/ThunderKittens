import torch 
import time
import torch.nn.functional as F
from einops import rearrange
import numpy as np

try:
    from mamba2.baselines.ssd_minimal import ssd_minimal_discrete 
except:
    from baselines.ssd_minimal import ssd_minimal_discrete

try:
    import thunderkittens as tk 
    print(f"Successfully imported thunderkittens")
except ImportError:
    print(f'Failed to import thunderkittens; Please "pip setup.py install".')
    tk = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print(f"Successfully imported mamba_chunk_scan_combined")
except ImportError:
    print(f"Failed to import mamba_chunk_scan_combined; Please 'pip install mamba_ssm'.")
    mamba_chunk_scan_combined = None


################### Mamba 2 ####################


def get_flops(b: int, n: int, dv: int, h: int, causal: bool = False) -> int:
    """Calculate FLOPs for Mamba-2 operation.
    
    Args:
        b: Batch size
        n: Sequence length
        dv: Head dimension
        h: Number of heads
        causal: Whether using causal attention
    
    Returns:
        Total number of FLOPs
    """
    # Constants
    chunk_length = 64
    state_dimension = 64
    ngroups = 1
    
    num_chunks = n // chunk_length
    seqlen = n
    nheads = h
    head_dimension = dv
    
    flops = 0
    
    # Mask and decay computations
    flops += seqlen * nheads  # cumsum
    flops += seqlen * nheads * chunk_length  # segsum
    flops += seqlen * nheads * chunk_length  # exp
    
    # Center blocks
    flops += 2 * num_chunks * ngroups * (chunk_length * chunk_length * state_dimension)  # QK^T
    flops += num_chunks * nheads * (chunk_length * chunk_length)  # L mask
    flops += 2 * num_chunks * nheads * (chunk_length * chunk_length * head_dimension)  # V mult
    
    # Low-rank factors
    flops += num_chunks * nheads * chunk_length * 2  # decay states
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)  # state computation
    
    # Inter-chunk operations
    flops += nheads * num_chunks * num_chunks * 2  # chunk decay
    flops += num_chunks * nheads * head_dimension * state_dimension  # state update
    
    # Output conversion
    flops += nheads * num_chunks * chunk_length  # decay out
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)  # final output
    flops += num_chunks * nheads * chunk_length * head_dimension  # state multiply
    
    # Final addition
    flops += num_chunks * chunk_length * nheads * head_dimension
    
    return b * flops


def get_inputs_mamba(dtype: torch.dtype, b: int, h: int, n: int, dv: int, verbose: bool = True):
    """Generate inputs for Mamba-2 computation.
    
    Args:
        dtype: Data type for tensors
        b: Batch size
        h: Number of heads
        n: Sequence length
        dv: Head dimension
        verbose: Whether to print dimension info
    
    Returns:
        Tuple of input tensors and parameters
    """
    torch.manual_seed(42)
    
    batch, seqlen = b, n
    chunk_size = 64
    dim = h * dv
    headdim = dv
    ngroups = 1
    dstate = 64
    device = "cuda"
    
    if verbose:
        print(f"Dimensions: batch={batch}, seqlen={seqlen}, heads={h}, dim={dim}, headdim={headdim}")
    
    x = torch.randn(batch, seqlen, h, headdim, dtype=torch.float32, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, h, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = -torch.exp(torch.rand(h, dtype=torch.float32, device=device)).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(h, dtype=torch.float32, device=device)
    
    return x, dt, A, B, C, D, chunk_size, torch.float32


def run_triton(dt, b, h, n, dv, verbose=True, **kwargs):
    x, dt, A, B, C, D, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    
    torch.cuda.synchronize()
    start_events[0].record()

    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    
    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    return y, tot


def run_torch(dt, b, h, n, dv, verbose=True, **kwargs):
    x, dt, A, B, C, D, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    torch.cuda.synchronize()
    start_events[0].record()

    y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, block_len=chunk_size)

    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    
    return y_min, tot


def run_tk(dtype: torch.dtype, b: int, h: int, n: int, dv: int, 
           verbose: bool = True, num_warmup: int = 3, num_repeats: int = 1000):
    """Run Thunderkittens implementation with timing.
    
    Args:
        dtype: Data type for computation
        b, h, n, dv: Dimension parameters
        verbose: Whether to print progress
        num_warmup: Number of warmup runs
        num_repeats: Number of timed runs
    
    Returns:
        Tuple of (output tensor, average runtime in ms)
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    time.sleep(1)

    x, dt, A, B, C, D, chunk_size, _ = get_inputs_mamba(dtype, b, h, n, dv, verbose)
    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(x*dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(A*dt, "b h l -> b l h").to(torch.float32).contiguous().requires_grad_()

    print(f"{torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), torch.isnan(a).any()}\n")
    
    # Warmup runs
    for _ in range(num_warmup):
        torch.cuda.synchronize()
        
        with torch.no_grad():
            out_warmup = tk.mamba2(q, k, v, a)
        torch.cuda.synchronize()

        # Verify output
        vals_gt_1e15 = torch.sum(out_warmup > 1e15)
        if torch.isnan(out_warmup).any() or torch.isinf(out_warmup).any() or vals_gt_1e15 > 0:
            num_nans = torch.sum(torch.isnan(out_warmup))
            print(f"Invalid values detected in warmup output, num_nans={num_nans}, vals_gt_1e15={vals_gt_1e15}")
        else: 
            print(f"out_warmup mean: {out_warmup.mean()}") #-- {q.mean()} -- {k.mean()} -- {v.mean()} -- {a.mean()}")

    print("\nWarmup complete\n")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
            
    # Timed runs
    times = []
    for i in range(num_repeats):

        # # modification 
        # x, dt, A, B, C, D, chunk_size, _ = get_inputs_mamba(dtype, b, h, n, dv, verbose)
        # q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        # k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        # v = rearrange(x*dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
        # a = rearrange(A*dt, "b h l -> b l h").to(torch.float32).contiguous().requires_grad_()
            
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            output = tk.mamba2(q, k, v, a)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
        # Verify output
        vals_gt_1e15 = torch.sum(output > 1e15)
        if torch.isnan(output).any() or torch.isinf(output).any() or vals_gt_1e15 > 0:
            num_nans = torch.sum(torch.isnan(output))
            raise RuntimeError(f"Invalid values detected in output at iteration {i}, num_nans={num_nans}, vals_gt_1e15={vals_gt_1e15}")
            breakpoint()

    torch.cuda.empty_cache()
    
    avg_time = np.mean(times)
    if verbose:
        print(f"Average runtime out of {len(times)}: {avg_time:.2f} ms")

    print("-----------" * 10)
    
    return output, avg_time

    
IMPLEMENTATIONS = {
    # "mamba2_triton": run_triton,
    # "mamba2_torch": run_torch,
    "mamba2_tk": run_tk,
}

