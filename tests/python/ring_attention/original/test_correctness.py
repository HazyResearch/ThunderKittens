# from functools import partial
# import os

# import jax
# import jax.numpy as jnp
# from jax import make_mesh
# from jax.experimental.shard_map import shard_map
# from jax.sharding import PartitionSpec as PS

# from ringattention import ringattention, blockwise_feedforward


# # Params
# batch = 1
# seqlen = 512
# num_heads = 8
# head_dim = 64


# # Inputs (random, arbitrary values)
# xq = jnp.ones((batch, seqlen, num_heads, head_dim))
# xk = jnp.ones_like(xq)
# xv = jnp.ones_like(xq)
# attention_bias = jnp.zeros((batch, 1, 1, seqlen))
# segment_ids = jnp.zeros((batch, seqlen), dtype=jnp.int32)


# ring_attention_sharded = shard_map(
#     partial(
#         ringattention,
#         axis_name="sp",
#         float32_logits=True,
#         cache_idx=None,
#         blockwise_kwargs=dict(
#             causal_block_size=1,
#             deterministic=True, # or false
#             dropout_rng=None, # or other value
#             attn_pdrop=0.0, # or other value
#             query_chunk_size=512, # or other value
#             key_chunk_size=512, # or other value
#             dtype=jax.numpy.float32, # or other value
#             policy=jax.checkpoint_policies.nothing_saveable,
#             precision=None, # or other value
#             prevent_cse=True, # or other value
#         )
#     ),
#     mesh=jax.make_mesh((1, 1, 1, 8), ("dp", "fsdp", "sp", "tp")),
#     in_specs=(
#         PS(("dp", "fsdp"), "sp", "tp", None),
#         PS(("dp", "fsdp"), "sp", "tp", None),
#         PS(("dp", "fsdp"), "sp", "tp", None),
#         PS(("dp", "fsdp"), None, None, None),
#         PS(("dp", "fsdp"), None),
#     ),
#     out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
#     check_rep=False
# )

# attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)

import jax
import jax.numpy as jnp
import numpy as np
import torch

def xavier_uniform_np(*shape):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

# Parameters
B = 8 # batch size
S = 64 # sequence length
H = 16 # number of heads
D = 128 # embedding dimension
assert D % H == 0, 'D must be divisible by H'

# Make sure these match
torch_dtype = torch.bfloat16
jax_dtype = jnp.bfloat16
torch_device = torch.device('cuda:0')
jax_device = jax.devices('gpu')[0]

# Deterministic inputs
np.random.seed(0)

# Inputs
q = np.random.randn(B, S, D)
k = np.random.randn(B, S, D)
v = np.random.randn(B, S, D)
q_torch = torch.from_numpy(q).to(device=torch_device, dtype=torch_dtype)
k_torch = torch.from_numpy(k).to(device=torch_device, dtype=torch_dtype)
v_torch = torch.from_numpy(v).to(device=torch_device, dtype=torch_dtype)
q_jax = jax.device_put(jnp.array(q, dtype=jax_dtype), device=jax_device)
k_jax = jax.device_put(jnp.array(k, dtype=jax_dtype), device=jax_device)
v_jax = jax.device_put(jnp.array(v, dtype=jax_dtype), device=jax_device)

# Weights
W_q = xavier_uniform_np(D, D) # d_model x d_k
W_k = xavier_uniform_np(D, D) # d_model x d_k
W_v = xavier_uniform_np(D, D) # d_model x d_v
W_o = xavier_uniform_np(D, D) # H * d_v x d_model
W_i_torch = torch.from_numpy(
    np.concatenate([W_q, W_k, W_v], axis=0)
).to(device=torch_device, dtype=torch_dtype)
W_o_torch = torch.from_numpy(W_o).to(device=torch_device, dtype=torch_dtype)
W_q_jax = jax.device_put(jnp.array(W_q, dtype=jax_dtype), device=jax_device)
W_k_jax = jax.device_put(jnp.array(W_k, dtype=jax_dtype), device=jax_device)
W_v_jax = jax.device_put(jnp.array(W_v, dtype=jax_dtype), device=jax_device)
W_o_jax = jax.device_put(jnp.array(W_o, dtype=jax_dtype), device=jax_device)

# Pure MHA - Torch
torch_mha = torch.nn.MultiheadAttention(
    embed_dim=D,
    num_heads=H,
    dropout=0.0, # output determinism
    bias=False, 
    add_bias_kv=False, 
    add_zero_attn=False, 
    kdim=None, 
    vdim=None, 
    batch_first=True, # Q, K, V are (batch, seq, feature)
    device=torch_device,
    dtype=torch_dtype
)
with torch.no_grad(): # set weights, prevent grad tracking
    torch_mha.in_proj_weight.copy_(W_i_torch)
    torch_mha.out_proj.weight.copy_(W_o_torch)

# Pure MHA - Jax
def jax_mha(q, k, v, W_q, W_k, W_v, W_o):
    q_proj = jnp.matmul(q, W_q.T)
    k_proj = jnp.matmul(k, W_k.T)
    v_proj = jnp.matmul(v, W_v.T)

    d_h = D // H
    q_heads = q_proj.reshape(B, S, H, d_h).transpose(0, 2, 1, 3)
    k_heads_T = k_proj.reshape(B, S, H, d_h).transpose(0, 2, 3, 1)
    v_heads = v_proj.reshape(B, S, H, d_h).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale = 1.0 / jnp.sqrt(d_h)
    attn_logits = jnp.matmul(q_heads, k_heads_T) * scale
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    attn_out = jnp.matmul(attn_weights, v_heads)

    # Combine heads
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

    # Final linear projection
    out = jnp.matmul(attn_out, W_o.T)
    return out, attn_weights

# MHA forward pass
out_torch, attn_weights_torch = torch_mha(
    query=q_torch,
    key=k_torch, 
    value=v_torch,
    key_padding_mask=None,
    need_weights=True, # false for performance
    attn_mask=None,
    average_attn_weights=False, # relevant if need_weights=True
    is_causal=False
)

out_jax, jax_weights_torch = jax_mha(
    q_jax,
    k_jax,
    v_jax,
    W_q_jax,
    W_k_jax,
    W_v_jax,
    W_o_jax
)

# Compare outputs. Output shape is (batch, seq, feature)
TOL = 5e-2 # large due to bf16
out_torch_np = out_torch.detach().to(dtype=torch.float32, device='cpu').numpy()
out_jax_np = np.array(jax.device_get(out_jax)).astype(np.float32)
max_error = np.max(np.abs(out_torch_np - out_jax_np))
num_errors = np.sum(np.abs(out_torch_np - out_jax_np) > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_torch_np.size} (TOL: {TOL})')

# Print first rows
# print('Torch output:')
# print(out_torch_np[0, 0, :10])
# print('Jax output:')
# print(out_jax_np[0, 0, :10])
