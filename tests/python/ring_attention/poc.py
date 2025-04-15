# Proof of Concept for the ring attention implementation

from functools import partial
from time import time
import gc

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
import numpy as np
import torch

from implementations import generate_mha_inputs, mha_torch, ring_mha_torch
from original_ring_attention import ring_attention
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
_out_torch_ring = ring_mha_torch(Q, K, V, causal, num_devices=NUM_DEVICES)
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
del _out_ref, _out_torch_ring
del Q, K, V, dL
gc.collect()

# Real ring attention on NUM_DEVICE devices with ThunderKittens
print('Generating inputs for TK ring attention...')
Qs, Ks, Vs, dLs = generate_mha_inputs(
    B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES
)

print('Running TK ring attention...')
for _ in range(2):
    _out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal) # warmup
num_iters = 25
times = []
for _ in range(num_iters):
    start = time()
    _out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal)
    end = time()
    times.append(end - start)

# Average time for topology-unaware version: 345~349 ms
# Average time for topology-aware version: 344~346 ms
avg_ms = (sum(times) / num_iters) * 1000
print(f'Average time for TK ring attention: {avg_ms:.4f} ms')

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

# Clean up
for i in range(NUM_DEVICES - 1, -1, -1):
    torch.cuda.set_device(i)
    torch.cuda.synchronize(i)
    torch.cuda.empty_cache()
    del Qs[i], Ks[i], Vs[i], dLs[i], _out_tk_ring[i]
del Qs, Ks, Vs, dLs, _out_tk_ring
gc.collect()

# Original ring attention on NUM_DEVICE devices
print('Generating inputs for the original ring attention...')
Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='ring_mha_orig', num_devices=NUM_DEVICES)

print('Running the original ring attention...')
mesh = jax.make_mesh((1, 1, NUM_DEVICES, 1), ("dp", "fsdp", "sp", "tp"))
QKVO_ps = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
bias_ps = PartitionSpec(("dp", "fsdp"), None, None, None)
seg_ids_ps = PartitionSpec(("dp", "fsdp"), None)
ring_attn_sharded = shard_map( # shard_map automatically JITs the function
    partial(
        ring_attention,
        axis_name="sp",
        float32_logits=False,
        cache_idx=None,
        blockwise_kwargs=dict(
            causal_block_size=None, # no causal mask
            deterministic=True, # or false
            dropout_rng=None, # or other value
            attn_pdrop=0.0, # or other value
            query_chunk_size=N//8, # or other value
            key_chunk_size=N//8, # or other value
            dtype=jax.numpy.bfloat16, # or other value
            policy=jax.checkpoint_policies.nothing_saveable,
            precision=None, # or other value
            prevent_cse=True, # or other value
        )
    ),
    mesh=mesh,
    in_specs=(
        QKVO_ps,
        QKVO_ps,
        QKVO_ps,
        bias_ps,
        seg_ids_ps,
    ),
    out_specs=QKVO_ps,
    check_rep=False
)
Q = jax.device_put(Q.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
K = jax.device_put(K.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
V = jax.device_put(V.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))

for _ in range(2):
    _out_orig_ring = ring_attn_sharded(Q, K, V, attn_bias, seg_ids) # warmup
num_iters = 10
times = []
for _ in range(num_iters):
    start = time()
    _out_orig_ring = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    end = time()
    times.append(end - start)

avg_ms = (sum(times) / num_iters) * 1000
print(f'Average time for original ring attention: {avg_ms:.4f} ms')

out_orig_ring = _out_orig_ring.transpose(0, 2, 1, 3) # back to (batch, seq, head, feature)
out_orig_ring = np.array(jax.device_get(out_orig_ring)).astype(np.float32)

# Verify correctness
TOL = 1e-2 # large due to bf16
assert(out_ref.shape == out_orig_ring.shape)
assert(out_ref.dtype == out_orig_ring.dtype)
abs_diff = np.abs(out_orig_ring - out_ref)
max_error = np.max(abs_diff)
num_errors = np.sum(abs_diff > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_orig_ring.size} (TOL: {TOL})')

# Clean up
del Q, K, V, dL
del attn_bias, seg_ids, _out_orig_ring
gc.collect()
