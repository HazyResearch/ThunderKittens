from functools import cache, partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
import numpy as np
import torch

from original_ring_attention import ring_attention # original authors' public implementation
from tk_ring_attention import pgl_tensor

@cache # avoid recomputing inputs
def __generate_mha_inputs(B, H, N, D_h):
    np.random.seed(42)
    Q = np.random.randn(B, H, N, D_h)
    K = np.random.randn(B, H, N, D_h)
    V = np.random.randn(B, H, N, D_h)
    dL = np.random.randn(B, H, N, D_h)
    return Q, K, V, dL

def generate_mha_inputs(B, H, N, D_h, dtype='bf16', target='mha_torch', num_devices=1):
    '''Generates MHA inputs (batch, head, seq, feature) for the specified target.
       Uses the same random seed every time, so the inputs are deterministic as long as 
       the given dimensions are the same.'''

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

    Q, K, V, dL = __generate_mha_inputs(B, H, N, D_h)

    if target == 'mha_torch':
        torch_device = torch.device('cuda:0')
        Q = torch.tensor(Q, dtype=torch_dtype, device=torch_device, requires_grad=True)
        K = torch.tensor(K, dtype=torch_dtype, device=torch_device, requires_grad=True)
        V = torch.tensor(V, dtype=torch_dtype, device=torch_device, requires_grad=True)
        dL = torch.tensor(dL, dtype=torch_dtype, device=torch_device, requires_grad=False) # no grad tracking for dL/dO
        return Q, K, V, dL

    elif target == 'mha_jax': 
        jax_device = jax.devices('gpu')[0]
        Q = jax.device_put(jnp.array(Q, dtype=jax_dtype), device=jax_device)
        K = jax.device_put(jnp.array(K, dtype=jax_dtype), device=jax_device)
        V = jax.device_put(jnp.array(V, dtype=jax_dtype), device=jax_device)
        dL = jax.device_put(jnp.array(dL, dtype=jax_dtype), device=jax_device)
        return Q, K, V, dL

    elif target == 'ring_mha_orig': # user must transpose, shard, and put on device afterwards
        Q = jnp.array(Q, dtype=jax_dtype)
        K = jnp.array(K, dtype=jax_dtype)
        V = jnp.array(V, dtype=jax_dtype)
        dL = jnp.array(dL, dtype=jax_dtype)
        return Q, K, V, dL
    
    elif target == 'ring_mha_tk':
        if num_devices <= 1:
            raise ValueError('num_devices must be greater than 1 for ring_mha_tk')
        if N % num_devices != 0:
            raise ValueError('N must be divisible by num_devices for ring_mha_tk')
        Qs = []
        Ks = []
        Vs = []
        dLs = []
        device_ids = [i for i in range(num_devices)]
        for i, (q, k, v, dl) in enumerate(zip(
            np.split(Q, num_devices, axis=2), 
            np.split(K, num_devices, axis=2), 
            np.split(V, num_devices, axis=2), 
            np.split(dL, num_devices, axis=2)
        )):
            Qs.append(pgl_tensor(torch.tensor(q, dtype=torch_dtype, requires_grad=True), device_ids, i))
            Ks.append(pgl_tensor(torch.tensor(k, dtype=torch_dtype, requires_grad=True), device_ids, i))
            Vs.append(pgl_tensor(torch.tensor(v, dtype=torch_dtype, requires_grad=True), device_ids, i))
            dLs.append(pgl_tensor(torch.tensor(dl, dtype=torch_dtype, requires_grad=False), device_ids, i)) # no grad tracking for dL/dO
        return Qs, Ks, Vs, dLs

    else:
        raise ValueError(f'Invalid target: {target}')


def mha_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool):
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


def ring_mha_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool, num_devices: int):
    '''Ring attention on single device with PyTorch. Q, K, V are already projected, unsharded.

       Q: (batch, head, seq, feature)
       K: (batch, head, seq, feature)
       V: (batch, head, seq, feature)

       Returns: (batch, head, seq, feature)
    '''

    Qs = torch.chunk(Q, num_devices, dim=2)
    Ks = torch.chunk(K, num_devices, dim=2)
    Vs = torch.chunk(V, num_devices, dim=2)
    As = []

    num_QO_blocks = len(Qs)
    num_KV_blocks = len(Ks)

    for i in range(num_QO_blocks): 
        # "Outer loop". Done in parallel on `num_QO_blocks` devices
        # Qs[i] stay on device i, Ks[i] and Vs[i] are rotated

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
    out_torch_ring = torch.cat(As, dim=2) # (B, H, N * NUM_DEVICES, D_h)
    return out_torch_ring


def ring_mha_orig(Q: jax.Array, K: jax.Array, V: jax.Array, causal: bool, num_devices: int):
    '''The original ring attention implementation. Q, K, V are already projected.

       Q: (batch, head, seq, feature)
       K: (batch, head, seq, feature)
       V: (batch, head, seq, feature)

       Returns: (batch, head, seq, feature)
    '''

    B = Q.shape[0]
    N = Q.shape[2]

    mesh = jax.make_mesh((1, 1, num_devices, 1), ("dp", "fsdp", "sp", "tp"))
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
                causal_block_size=None, # no causal mask || TODO: support causal
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

    # Reshape and shard inputs accordingly
    # The original implementation uses (batch, seq, head, feature) format
    Q = jax.device_put(Q.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
    K = jax.device_put(K.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
    V = jax.device_put(V.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))

    # We don't care about bias / segments
    attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
    seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))

    # Calculate and reshape output
    out_ring_orig = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    out_ring_orig = out_ring_orig.transpose(0, 2, 1, 3) # back to (batch, seq, head, feature)

    return out_ring_orig
