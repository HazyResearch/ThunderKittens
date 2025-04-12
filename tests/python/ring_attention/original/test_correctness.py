from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS
import numpy as np
import torch
from ringattention import ringattention, blockwise_feedforward # original authors' public implementation


def generate_mha_inputs(B, H, N, D_h, dtype='bf16', num_devices=1):
    '''Generates MHA inputs (batch, head, seq, feature) for both Torch and JAX, using the same values'''

    if dtype == 'bf16':
        torch_dtype = torch.bfloat16
        jax_dtype = jnp.bfloat16
    elif dtype == 'fp16':
        torch_dtype = torch.float16
        jax_dtype = jnp.float16
    elif dtype == 'fp32':
        torch_dtype = torch.float32
        jax_dtype = jnp.float32
    else:
        raise ValueError(f'Invalid dtype: {dtype}')
    
    torch_device = torch.device('cuda:0')
    jax_device = jax.devices('gpu')[0]

    np.random.seed(42)
    Q = np.random.randn(B, H, N, D_h)
    K = np.random.randn(B, H, N, D_h)
    V = np.random.randn(B, H, N, D_h)
    dL = np.random.randn(B, H, N, D_h)
    Q_torch = torch.from_numpy(Q).to(device=torch_device, dtype=torch_dtype).requires_grad_(True)
    K_torch = torch.from_numpy(K).to(device=torch_device, dtype=torch_dtype).requires_grad_(True)
    V_torch = torch.from_numpy(V).to(device=torch_device, dtype=torch_dtype).requires_grad_(True)
    dL_torch = torch.from_numpy(dL).to(device=torch_device, dtype=torch_dtype) # no grad tracking for dL/dO
    Q_jax = jax.device_put(jnp.array(Q, dtype=jax_dtype), device=jax_device)
    K_jax = jax.device_put(jnp.array(K, dtype=jax_dtype), device=jax_device)
    V_jax = jax.device_put(jnp.array(V, dtype=jax_dtype), device=jax_device)
    dL_jax = jax.device_put(jnp.array(dL, dtype=jax_dtype), device=jax_device)

    return Q_torch, K_torch, V_torch, dL_torch, Q_jax, K_jax, V_jax, dL_jax


def mha_pytorch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool):
    '''Pytorch attention implementation. Q, K, V are already projected.

       Q: (batch, head, seq, feature)
       K: (batch, head, seq, feature)
       V: (batch, head, seq, feature)
       causal: bool

       Returns: (batch, head, seq, feature)
    '''

    QK = torch.matmul(Q, K.transpose(-2, -1))
    QK /= (Q.size(-1) ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(device=QK.device, dtype=torch.bool)
        QK.masked_fill_(mask, float('-inf'))
    attn_weights = torch.nn.functional.softmax(QK, dim=-1)
    attn_out = torch.matmul(attn_weights, V)

    return attn_out


def mha_jax(Q: jax.Array, K: jax.Array, V: jax.Array, causal: bool):
    '''JAX attention implementation. Q, K, V are already projected.

       Q: (batch, head, seq, feature)
       K: (batch, head, seq, feature)
       V: (batch, head, seq, feature)

       Returns: (batch, head, seq, feature)
    '''

    QK = jnp.matmul(Q, K.transpose(0, 1, 3, 2))
    QK /= (Q.shape[-1] ** 0.5)
    if causal:
        mask = jnp.triu(jnp.ones((QK.shape[-2], QK.shape[-1]), dtype=jnp.bool), k=1)
        QK = jnp.where(mask, jnp.asarray(-jnp.inf, dtype=QK.dtype), QK)
    attn_weights = jax.nn.softmax(QK, axis=-1)
    attn_out = jnp.matmul(attn_weights, V)

    return attn_out


# Parameters
B = 8 # batch size
S = 64 # sequence length
H = 16 # number of heads
D_h = 16 # head dimension
dtype = 'bf16'
causal = False

# Generate inputs
q_torch, k_torch, v_torch, dL_torch, q_jax, k_jax, v_jax, dL_jax = generate_mha_inputs(
    B, H, S, D_h, dtype=dtype
)

# Run MHAs
out_torch = mha_pytorch(q_torch, k_torch, v_torch, causal)
out_jax = mha_jax(q_jax, k_jax, v_jax, causal)

# Verify correctness. Output shape is (batch, seq, feature)
TOL = 1e-2 # large due to bf16
out_torch = out_torch.detach().to(dtype=torch.float32, device='cpu').numpy()
out_jax = np.array(jax.device_get(out_jax)).astype(np.float32)
max_error = np.max(np.abs(out_torch - out_jax))
num_errors = np.sum(np.abs(out_torch - out_jax) > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_torch.size} (TOL: {TOL})')

# Sanity check
# print('Torch output:')
# print(out_torch[0, 0, 0, :10])
# print('Jax output:')
# print(out_jax[0, 0, 0, :10])

# Ring Attention - Jax (original authors' implementation)
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

# # Inputs (random, arbitrary values)
# xq = jnp.ones((batch, seqlen, num_heads, head_dim))
# xk = jnp.ones_like(xq)
# xv = jnp.ones_like(xq)
# attention_bias = jnp.zeros((batch, 1, 1, seqlen))
# segment_ids = jnp.zeros((batch, seqlen), dtype=jnp.int32)

# attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)
