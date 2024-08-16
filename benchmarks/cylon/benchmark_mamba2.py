import sys
import torch
import time
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except: 
    print("Please run: pip install mamba_ssm")
    sys.exit(1)


torch.manual_seed(42)


def compute_flops(batch_size, chunk_length, num_chunks, head_dimension, nheads, state_dimension=64, ngroups=1):
    """
    Mamba-2 FLOPS Calculator 
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py 

    ngroups: default for multi-value attention 
    state_dimension: expansion factor for the recurrent state
    """

    flops = 0
    seqlen = chunk_length * num_chunks

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
    flops += 2 * ( num_chunks * nheads * chunk_size * head_dimension * state_dimension )

    # Compute the inter-chunk SSM recurrence
    # decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    flops += 1 * ( nheads * num_chunks * num_chunks ) # segsum
    flops += 1 * ( nheads * num_chunks * num_chunks ) # exp
    # new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    flops += 1 * ( num_chunks * nheads * head_dimension * state_dimension )

    # Compute state -> output conversion per chunk
    # state_decay_out = torch.exp(A_cumsum)
    flops += 1 * ( nheads * num_chunks * chunk_size ) # exp
    # Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    # -- step 1: C_adjusted, states_adjusted = C.transpose(2,3), states.transpose(3,4) 
    #            intermediate = torch.matmul(C_adjusted, states_adjusted)
    flops += 2 * ( num_chunks * nheads * chunk_size * head_dimension * state_dimension )
    # -- step 2: state_decay_adjusted = state_decay_out.transpose(1, 2)..unsqueeze(-1)
    #            Y_check = (intermediate * state_decay_adjusted).transpose(2,3)
    flops += 1 * ( num_chunks * nheads * chunk_size * head_dimension )

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y = Y_diag+Y_off
    flops += 1 * ( num_chunks * chunk_size * nheads * head_dimension )

    # batchify and return 
    flops = batch_size * flops
    return flops 


def reference_flops(batch_size, chunk_length, num_chunks, head_dimension, nheads, state_dimension=64):
    """ Sanity checking against Based total flop count """ 

    f = 0
    seqlen = chunk_length * num_chunks
    expanded_dim = 273

    f = 2 * batch_size * seqlen * nheads * expanded_dim # feature map
    # print(f"-- step 1: {2 * batch_size * seqlen * nheads * expanded_dim}")

    f += batch * seqlen * nheads * head_dimension * expanded_dim # (k * v)
    # print(f"-- step 2: {batch * seqlen * nheads * head_dimension * expanded_dim}")
    f += batch * seqlen * nheads * head_dimension * expanded_dim # cumsum 
    # print(f"-- step 3: {batch * seqlen * nheads * head_dimension * expanded_dim}")
    f += batch * seqlen * nheads * head_dimension * expanded_dim # (q * (k * v))
    # print(f"-- step 4: {batch * seqlen * nheads * head_dimension * expanded_dim}")
    f += batch * seqlen * nheads * head_dimension * expanded_dim # .sum(dim=-1)
    # print(f"-- step 5: { batch * seqlen * nheads * head_dimension * expanded_dim}")

    return f


def get_timings(x, dt, A, B, C, chunk_size, warmup_iters=10):
    
    for i in range(warmup_iters):
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None) 

    torch.cuda.synchronize()
    start = time.time()

    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None) 

    torch.cuda.synchronize()
    end = time.time()
    return end-start


# Construct inputs 
# Dimensions. Denoted (B, T, Q, D, P) in the paper
batch, seqlen, chunk_size, dim, headdim = 16, 2048, 64, 2048, 256
nheads = dim // headdim  # (H) in the paper
ngroups = 1 # (G) in the paper
dstate =  128 # (N) in the paper
dtype = torch.float32
device = "cuda"
num_chunks = seqlen // chunk_size

x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
D = torch.randn(nheads, dtype=dtype, device=device)


# Compute efficiency
flops = compute_flops(
    batch_size=batch, 
    chunk_length=chunk_size, 
    num_chunks=num_chunks, 
    head_dimension=headdim, 
    nheads=nheads, 
    state_dimension=dstate,
    ngroups=ngroups,
)
print(f"{flops=} # FLOPS for Mamba-2")
ref_flops = reference_flops(batch, chunk_size, num_chunks, nheads, headdim)
print(f"flops={ref_flops} # (Reference) FLOPS count for Based")
timings = get_timings(x, dt, A, B, C, chunk_size)

efficiency = (flops / 1e12) / (timings)  
print(f"{efficiency=} TFLOPS for Mamba-2 layer")


