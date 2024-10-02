import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from tqdm import trange
import numpy as np
import sys
import math

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    
    return Y, final_state

torch.manual_seed(0)

batch   = 1
headdim = 64

length     = int(sys.argv[1])
chunk_size = 64
num_chunks = length // chunk_size

d_head  = 64
n_heads = headdim // d_head
d_state = 64
ngroups = 1

# print hardcoded values
print("-----------------")
print("batch:", batch)
print("n_heads:", n_heads)
print("ngroups:", ngroups)
print("length:", length)
print("d_head:", d_head)
print("d_state:", d_state)
print("-----------------")

dtype  = torch.float32
device = 'cuda'

test_segsum = torch.arange(4, dtype=dtype, device=device)
test_segsum = segsum(test_segsum)

x  = torch.randn(batch, length, n_heads, d_head, dtype=dtype, device=device)
dt = F.softplus(torch.randn(batch, length, n_heads, dtype=torch.float32, device=device) - 4).requires_grad_()
A = (-torch.exp(torch.rand(n_heads, dtype=torch.float32, device=device))).requires_grad_()
B = torch.randn(batch, length, ngroups, d_state, dtype=dtype, device=device)
C = torch.randn(batch, length, ngroups, d_state, dtype=dtype, device=device)

X = x*dt.unsqueeze(-1)
A = A*dt
B = B
C = C

y_ref, _ = ssd_minimal_discrete(X, A, B, C, block_len=chunk_size)

X = X.permute(0, 2, 1, 3)
A = A.permute(0, 2, 1)
B = B.permute(0, 2, 1, 3)
C = C.permute(0, 2, 1, 3)
y_ref = y_ref.permute(0, 2, 1, 3)

# print shapes 
print("X:", X.shape)
print("A:", A.shape)
print("B:", B.shape)
print("C:", C.shape)
print("Y:", y_ref.shape)
print("-----------------")

filename = f'randn_{length}N_{d_head}D_{d_state}S.txt'

with open(filename, 'w') as f:
    
    # inputs
    xf = X.to(torch.float32).flatten().detach().cpu().numpy()
    Af = A.to(torch.float32).flatten().detach().cpu().numpy()
    Bf = B.to(torch.float32).flatten().detach().cpu().numpy()
    Cf = C.to(torch.float32).flatten().detach().cpu().numpy()
    
    yf = y_ref.to(torch.float32).flatten().detach().cpu().numpy()
    
    print(f'X (V) shape: {X.shape}, total elements {X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]}')
    print(f'A (alpha) shape: {A.shape}, total elements {A.shape[0] * A.shape[1] * A.shape[2]}')
    print(f'B (K) shape: {B.shape}, total elements {B.shape[0] * B.shape[1] * B.shape[2] * B.shape[3]}')
    print(f'C (Q) shape: {C.shape}, total elements {C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3]}')
    print(f'Y (O) shape: {y_ref.shape}, total elements {y_ref.shape[0] * y_ref.shape[1] * y_ref.shape[2] * y_ref.shape[3]}')

    print((C[0,0]@B[0,0].T * torch.exp(segsum(A[0,0]))) @ X[0,0])
    print(y_ref[0,0])
    # print(torch.exp(segsum(A[0,0])))

    for i in trange(X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]):
        f.write(repr(xf[i])) # Q
        f.write(' ')
    for i in trange(A.shape[0] * A.shape[1] * A.shape[2]):
        f.write(repr(Af[i])) # A
        f.write(' ')
    for i in trange(B.shape[0] * B.shape[1] * B.shape[2] * B.shape[3]):
        f.write(repr(Bf[i])) # K
        f.write(' ')
    for i in trange(C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3]):
        f.write(repr(Cf[i])) # V
        f.write(' ')
    for i in trange(y_ref.shape[0] * y_ref.shape[1] * y_ref.shape[2] * y_ref.shape[3]):
        f.write(repr(yf[i]))
        f.write(' ')








