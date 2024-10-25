import torch 
import time
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from mamba2.baselines.ssd_minimal import ssd_minimal_discrete 

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


def get_flops(b, n, dv, h, causal=False):
    """
    Mamba-2 FLOPS Calculator 
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py 

    ngroups: default for multi-value attention 
    state_dimension: expansion factor for the recurrent state
    """
    dt = torch.bfloat16
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv, verbose=False)
    batch_size = x.shape[0]
    seqlen = x.shape[1]
    nheads = x.shape[2]
    head_dimension = x.shape[-1]
    chunk_length=chunk_size
    state_dimension=B.shape[-1]
    ngroups=B.shape[-2]

    flops = 0
    # seqlen = chunk_length * num_chunks
    num_chunks = seqlen // chunk_length

    # construct mask with decays
    # A_cumsum = torch.cumsum(A, dim=-1)
    # L = torch.exp(segsum(A))
    flops += 1 * ( seqlen * nheads ) # cumsum 
    flops += 1 * ( seqlen * nheads * chunk_length ) # segsum
    flops += 1 * ( seqlen * nheads * chunk_length ) # exp

    # compute center blocks 
    # Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    flops += 2 * num_chunks * ngroups * ( chunk_length * chunk_length * state_dimension )  # mult. QK^T
    flops += 1 * num_chunks * nheads  * ( chunk_length * chunk_length )                    # mask with L
    flops += 2 * num_chunks * nheads  * ( chunk_length * chunk_length * head_dimension)    # mult. by V

    #### low-rank factors ###
    # decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    flops += 1 * ( num_chunks * nheads * chunk_length ) # subtract
    flops += 1 * ( num_chunks * nheads * chunk_length ) # exp

    # Compute the state for each intra-chunk
    # states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # -- mult1 = K * rearrange(decay_states, "b h c l -> b c l h").unsqueeze(-1)
    flops += 1 * ( num_chunks * chunk_length *  nheads )
    # -- mult2 = torch.einsum("bclhn,bclhp->bchpn", mult1, V)
    flops += 2 * ( num_chunks * nheads * chunk_length * head_dimension * state_dimension )

    # Compute the inter-chunk SSM recurrence
    # decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    flops += 1 * ( nheads * num_chunks * num_chunks ) # segsum
    flops += 1 * ( nheads * num_chunks * num_chunks ) # exp
    # new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    flops += 1 * ( num_chunks * nheads * head_dimension * state_dimension )

    # Compute state -> output conversion per chunk
    # state_decay_out = torch.exp(A_cumsum)
    flops += 1 * ( nheads * num_chunks * chunk_length ) # exp
    # Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    # -- step 1: C_adjusted, states_adjusted = C.transpose(2,3), states.transpose(3,4) 
    #            intermediate = torch.matmul(C_adjusted, states_adjusted)
    flops += 2 * ( num_chunks * nheads * chunk_length * head_dimension * state_dimension )
    # -- step 2: state_decay_adjusted = state_decay_out.transpose(1, 2)..unsqueeze(-1)
    #            Y_check = (intermediate * state_decay_adjusted).transpose(2,3)
    flops += 1 * ( num_chunks * nheads * chunk_length * head_dimension )

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y = Y_diag+Y_off
    flops += 1 * ( num_chunks * chunk_length * nheads * head_dimension )

    # batchify and return 
    flops = batch_size * flops
    return flops 


def get_inputs_mamba(dt, b, h, n, dv, verbose=True, **kwargs):
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = b, n, 64, (h*dv), 64
    # print(f"FLAG: {batch=}, {seqlen=}, {chunk_size=}, {dim=}, {headdim=}")
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x  = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)
    z = None

    return x, dt, 0, A, B, C, D, z, chunk_size, dtype


def run_triton(dt, b, h, n, dv, verbose=True, **kwargs):
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    
    torch.cuda.synchronize()
    start_events[0].record()

    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    
    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    return y, tot

def run_torch(dt, b, h, n, dv, verbose=True, **kwargs):
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    torch.cuda.synchronize()
    start_events[0].record()

    D_factor = x * rearrange(D, "d -> d 1")
    _dt = F.softplus(dt + dt_bias)
    y_min, _ = ssd_minimal_discrete(x*_dt.unsqueeze(-1), A*_dt, B, C, block_len=chunk_size)
    y_min += D_factor

    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    
    return y_min, tot

def run_tk(dt, b, h, n, dv, verbose=True, **kwargs):
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba(dt, b, h, n, dv)

    torch.cuda.empty_cache()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

    # TK 
    D_factor = x * rearrange(D, "d -> d 1")
    _dt = F.softplus(dt + dt_bias)
    v = x*_dt.unsqueeze(-1)
    a = A*_dt
    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(v, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(a, "b h l -> b l h").to(torch.float32).contiguous()

    torch.cuda.synchronize()
    start_events[0].record()

    y_tk = tk.mamba2(q, k, v, a) 
    
    # breakpoint()

    end_events[0].record()
    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]

    # 10 TFLOPS in these operations
    y_tk = rearrange(y_tk, "b h l d -> b l h d").contiguous()
    y_tk += D_factor
    
    if np.isnan(y_tk.float().cpu()).any() or np.isinf(y_tk.float().cpu()).any():
        breakpoint()
    assert not np.isnan(y_tk.float().cpu()).any(), "NaN values detected in output 'y_tk'"
    assert not np.isinf(y_tk.float().cpu()).any(), "Inf values detected in output 'y_tk'"
    return y_tk, tot

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

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(1)]

        torch.cuda.synchronize()
        start_events[0].record()

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

        end_events[0].record()
        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)][0]
    except Exception as e:
        tot = -1
        y = None
    return y, tot

    
IMPLEMENTATIONS = {
    # "mamba2_triton": run_triton,
    "mamba2_torch": run_torch,
    "mamba2_tk": run_tk,
}

