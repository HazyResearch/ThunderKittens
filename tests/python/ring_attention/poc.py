# Proof of Concept
# Step-by-step implementation of what should be calculated at each step of the ring attention

from time import time
import gc

import numpy as np
import torch

from implementations import (
    generate_mha_inputs,
    mha_torch,
    ring_mha_orig
)
from tk_ring_attention import ring_mha_forward

# Parameters
NUM_DEVICES = 8
B = 16 # batch size
H = 16 # number of heads
N = 768 * NUM_DEVICES # sequence length (must be 768 * (2**i))
D_h = 64 # head dimension (DO NOT CHANGE)
dtype = 'bf16'
causal = False

assert NUM_DEVICES >= 1, 'NUM_DEVICES must be >= 1'
assert N % NUM_DEVICES == 0, 'N must be divisible by NUM_DEVICES'

# Generate inputs
print('Generating inputs...')
torch.cuda.set_device(0)
Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='mha_torch', num_devices=1)

# Calculate reference output
print('Running reference MHA...')
_out_ref = mha_torch(Q, K, V, causal)
out_ref = _out_ref.detach().to(dtype=torch.float32, device='cpu').numpy()

# Split inputs, but on device 0 (for PoC with PyTorch)
torch.cuda.set_device(0) # just in case
Qs = torch.chunk(Q, NUM_DEVICES, dim=2)
Ks = torch.chunk(K, NUM_DEVICES, dim=2)
Vs = torch.chunk(V, NUM_DEVICES, dim=2)
dLs = torch.chunk(dL, NUM_DEVICES, dim=2)

# Ring attention on single device with PyTorch
print('Running simulated ring attention...')
As = []
num_QO_blocks = len(Qs)
num_KV_blocks = len(Ks)
for i in range(num_QO_blocks): 
    # "Outer loop". Done in parallel on `num_QO_blocks` devices
    # Qs[i] stay on device i, Ks[i] and Vs[i] are rotated
    torch.cuda.set_device(0) # just in case

    # We only need to scale once
    Qi = Qs[i] / (Qs[i].size(-1) ** 0.5)

    # Accumulating variables
    numerator = torch.zeros_like(Qi) # (B, H, N, D_h), N = block_len
    denominator = torch.zeros(
        Qi.shape[:-1], dtype=Qi.dtype, device=Qi.device, layout=Qi.layout
    ) # (B, H, N)
    local_max = torch.full(
        denominator.shape, float('-inf'), dtype=Qi.dtype, device=Qi.device, layout=Qi.layout
    ) # (B, H, N)

    for rotation_idx in range(num_KV_blocks):
        # "Inner loop". Done sequentially on each device. 
        # `num_KV_blocks` ring rotations of Ks and Vs

        # device i starts with Ks[i] and Vs[i]
        j = (i + rotation_idx) % num_KV_blocks
        Kj = Ks[j]
        Vj = Vs[j]

        # Blockwise attention
        QiKj = torch.matmul(Qi, Kj.transpose(-1, -2)) # (B, H, N, N)
        new_max = torch.max(local_max, torch.max(QiKj, dim=-1).values) # (B, H, N)
        exp_QiKj = torch.exp(QiKj - new_max.unsqueeze(-1)) # (B, H, N, N)
        if rotation_idx > 0:
            rescaler = torch.exp(local_max - new_max)
            numerator *= rescaler.unsqueeze(-1)
            denominator *= rescaler
        numerator += torch.matmul(exp_QiKj, Vj)
        denominator += torch.sum(exp_QiKj, dim=-1)
        local_max = new_max

    # Normalize and store
    Ai = numerator / denominator.unsqueeze(-1) # (B, H, N, D_h)
    As.append(Ai)

# Concatenate and transfer to CPU
_out_torch_ring = torch.cat(As, dim=2) # (B, H, N * NUM_DEVICES, D_h)
out_torch_ring = _out_torch_ring.detach().to(dtype=torch.float32, device='cpu').numpy()

# Verify correctness
TOL = 1e-2 # large due to bf16
assert(out_ref.shape == out_torch_ring.shape)
assert(out_ref.dtype == out_torch_ring.dtype)
abs_diff = np.abs(out_torch_ring - out_ref)
max_error = np.max(abs_diff)
num_errors = np.sum(abs_diff > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_torch_ring.size} (TOL: {TOL})')

# Clean up
for i in range(NUM_DEVICES - 1, -1, -1):
    torch.cuda.set_device(i)
    torch.cuda.synchronize(i)
    torch.cuda.empty_cache()
    del As[i]
del As, Qs, Ks, Vs, dLs
del Ai, Qi, Kj, Vj, QiKj, exp_QiKj, rescaler
del numerator, denominator, local_max, new_max
del _out_ref, _out_torch_ring
del Q, K, V, dL
gc.collect()

# Real ring attention on NUM_DEVICE devices with ThunderKittens
print('Generating inputs for TK ring attention...')
Qs, Ks, Vs, dLs = generate_mha_inputs(
    B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES
)

print('Running TK ring attention...')
for _ in range(2): _ = ring_mha_forward(Qs, Ks, Vs, causal) # warmup
num_iters = 10
start = time()
for _ in range(num_iters): _ = ring_mha_forward(Qs, Ks, Vs, causal)
end = time()
print(f'Average time for TK ring attention: {1000 * (end - start) / num_iters:.4f} ms')

# Average time for topology-unaware version: 345~349 ms
# Average time for topology-aware version: ?

_out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal)
out_tk_ring = [t.detach().to(dtype=torch.float32, device='cpu').numpy() for t in _out_tk_ring]
out_tk_ring = np.concatenate(out_tk_ring, axis=2) # (B, H, N * NUM_DEVICES, D_h)

# Verify correctness
TOL = 1e-2 # large due to bf16
assert(out_ref.shape == out_tk_ring.shape)
assert(out_ref.dtype == out_tk_ring.dtype)
abs_diff = np.abs(out_tk_ring - out_ref)
max_error = np.max(abs_diff)
num_errors = np.sum(abs_diff > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_tk_ring.size} (TOL: {TOL})')
