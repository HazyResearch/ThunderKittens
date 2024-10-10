import torch 
import time
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
def get_inputs_mamba():
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 4, 2048, 64, 2048, 64
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
    return x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype


def run_triton():
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba()
    
    torch.cuda.synchronize()
    start = time.time()
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize()
    end = time.time()
    tot = end - start
    return tot

def run_torch():
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba()

    torch.cuda.synchronize()
    start = time.time()

    D_factor = x * rearrange(D, "d -> d 1")
    _dt = F.softplus(dt + dt_bias)
    y_min, _ = ssd_minimal_discrete(x*_dt.unsqueeze(-1), A*_dt, B, C, block_len=chunk_size)
    y_min += D_factor

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start
    return tot

def run_tk():
    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba()

    q = C.to(torch.bfloat16)
    k = B.to(torch.bfloat16)
    v = x.to(torch.bfloat16)

    # 4 TFLOPS in these multiplies
    D_factor = (x * rearrange(D, "d -> d 1")).to(torch.bfloat16)
    _dt = F.softplus(dt + dt_bias)
    v = v*_dt.unsqueeze(-1).to(torch.bfloat16)
    a = A*_dt

    # 40 TFLOPS in these transposes
    q = rearrange(q, "b l h d -> b h l d").contiguous()
    k = rearrange(k, "b l h d -> b h l d").contiguous()
    v = rearrange(v, "b l h d -> b h l d").contiguous()
    a = rearrange(a, "b h l -> b l h").contiguous()

    torch.cuda.synchronize()
    start = time.time()

    y_tk = tk.mamba2(q, k, v, a) 

    torch.cuda.synchronize()
    end = time.time()
    tot = end - start

    # 10 TFLOPS in these operations
    y_tk = rearrange(y_tk, "b h l d -> b l h d").contiguous()
    y_tk += D_factor
    
    return tot


def get_flops_mamba2():
    """
    Mamba-2 FLOPS Calculator 
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py 

    ngroups: default for multi-value attention 
    state_dimension: expansion factor for the recurrent state
    """

    x, dt, dt_bias, A, B, C, D, z, chunk_size, dtype = get_inputs_mamba()
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


def efficiency(flops, time):
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms

if __name__ == "__main__":
    num_warmup = 2
    num_repeats = 5

    for fn in [run_triton, run_torch, run_tk]:
        for _ in range(num_warmup): fn()
        _time = 0
        for _ in range(num_repeats): _time += fn()
        flops = get_flops_mamba2()
        microseconds = ( _time * 1000000 ) / num_repeats
        eff = efficiency(flops, microseconds)
        print(f"{eff} TFLOPS/ms")



