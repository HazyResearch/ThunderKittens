from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
import numpy as np
import torch
from ringattention import ring_attention # original authors' public implementation


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
    '''Pytorch MHA implementation. Q, K, V are already projected.

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
    '''JAX MHA implementation. Q, K, V are already projected.

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
H = 16 # number of heads
N = 1024 # sequence length
D_h = 32 # head dimension
dtype = 'bf16'
causal = False

# Generate inputs
Q_torch, K_torch, V_torch, dL_torch, Q_jax, K_jax, V_jax, dL_jax = generate_mha_inputs(
    B, H, N, D_h, dtype=dtype
)

# Run MHAs
out_torch = mha_pytorch(Q_torch, K_torch, V_torch, causal)
out_jax = mha_jax(Q_jax, K_jax, V_jax, causal)

# Ring Attention - Jax (original authors' implementation)
mesh = jax.make_mesh((1, 1, 8, 1), ("dp", "fsdp", "sp", "tp")) # todo use num_devices
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
# (batch, seq, head, feature)
Q = jax.device_put(Q_jax.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
K = jax.device_put(K_jax.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
V = jax.device_put(V_jax.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
# we don't use bias / segments
attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))
# Calculate
out_ring_orig = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
out_ring_orig = out_ring_orig.transpose(0, 2, 1, 3) # back to (batch, seq, head, feature)

# Verify correctness. Output shape is (batch, seq, feature)
TOL = 1e-2 # large due to bf16
out_torch = out_torch.detach().to(dtype=torch.float32, device='cpu').numpy()
out_jax = np.array(jax.device_get(out_jax)).astype(np.float32)
out_ring_orig = np.array(jax.device_get(out_ring_orig)).astype(np.float32)
max_error = np.max(np.abs(out_torch - out_ring_orig))
num_errors = np.sum(np.abs(out_torch - out_ring_orig) > TOL)
print(f'Max abs diff: {max_error}')
print(f'Num errors: {num_errors} out of {out_torch.size} (TOL: {TOL})')

