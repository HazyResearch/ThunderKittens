import numpy as np
import flax.linen as nn
from flax.linen import partitioning
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from einops import rearrange
from functools import partial
import dataclasses
import functools
from typing import Any, NamedTuple, Optional
import os


def _ring_attention_fwd(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs):
    '''q, k, v: (batch, seq_len, num_heads, dim_per_head)'''

    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape

    # Softmax accumulation
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype) # exp(Q_i @ K_j^T) @ V_j
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)             # sum(exp(Q_i @ K_j^T))
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype) # max(Q_ii @ K_jj^T)

    # Number of devices
    axis_size = lax.psum(1, axis_name)

    # Assume this function is pre-sharded inside shard_map; i.e., the given q, k, v are already sharded
    q_block_size, kv_block_size = q_len, kv_len 
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry

        if cache_idx is None:
            q_block_idx = lax.axis_index(axis_name) # should be same as device index
            # jax.debug.print("q_block_idx: {}", q_block_idx)
            q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        else:
            q_chunk_idx_start = cache_idx // query_chunk_size

        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)

        numerator, denominator, max_score = _blockwise_attention_fwd(
            q, k, v,
            (numerator, denominator, prev_max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            cache_idx=cache_idx,
            **blockwise_kwargs
        )

        # Send and receive the next block of K & V
        k, v = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]), 
            (k, v)
        )
        return (max_score, numerator, denominator, k, v), None

    # Perform the actual ring attention
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block, init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size)
    )

    # Perform the softmax normalization and return
    # rearrange here is equivalent to transpose(0, 2, 1)
    # [..., None] is equivalent to adding a new axis at the end. Becomes (B, Q, H, 1)
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_bias, segment_ids, cache_idx, denominator, max_score)

def _ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    output, q, k, v, attn_bias, segment_ids, cache_idx, denominator, max_score = res
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=q.dtype)
    dk = jnp.zeros_like(k, dtype=k.dtype)
    dv = jnp.zeros_like(v, dtype=k.dtype)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    q_block_size, kv_block_size = q_len, kv_len # assumes this function is pre-sharded inside shard_map
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        if cache_idx is None:
            q_block_idx = lax.axis_index(axis_name)
            q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        else:
            q_chunk_idx_start = cache_idx // query_chunk_size
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(
            q, k, v, g,
            (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            cache_idx=cache_idx,
            **blockwise_kwargs
        )
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(k.dtype)
    return dq, dk, dv, None, None, None

@partial(jax.custom_vjp, nondiff_argnums=[6, 7, 8])
def ring_attention(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_attention_fwd(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs)
    return y

ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)

def _blockwise_attention_fwd(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal_block_size, query_chunk_size,
                             key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse, cache_idx):
    
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape

    # Calculate the number of chunks per block
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size

    # Reshape q, k, v to chunked format
    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v)) # now (chunk_id, batch, chunk_size, num_heads, dim_per_head)

    # Recall that:
    #   numerator is (batch, block_size, num_heads, dim_per_head)
    #   denominator is (batch, num_heads, block_size)
    #   max_score is (batch, num_heads, block_size)
    # We reshape to:
    #   numerator (chunk_id, batch, chunk_size, num_heads, dim_per_head)
    #   denominator (chunk_id, batch, num_heads, chunk_size)
    #   max_score (chunk_id, batch, num_heads, chunk_size)
    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    scale = jnp.sqrt(q.shape[-1])

    # Dropout
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    
    # Define inner functions
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, segment_ids, deterministic,
        attn_dropout, attn_pdrop, causal_block_size, dtype)
    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(attn_weights, axis=-1))
            max_score_chunk = lax.stop_gradient(max_score_chunk)
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None])
            exp_values = jnp.einsum('bhqk,bkhd->bqhd', exp_weights, value_chunk, precision=precision)
            correction = rearrange(jnp.exp(prev_max_score_chunk - max_score_chunk), 'b h q -> b q h')[..., None]
            numerator_chunk = numerator_chunk * correction + exp_values
            denominator_chunk = denominator_chunk * jnp.exp(prev_max_score_chunk - max_score_chunk) + exp_weights.sum(axis=-1)
            return (numerator_chunk, denominator_chunk, max_score_chunk), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            should_run = jnp.array(True)
            if causal_block_size is not None:
                should_run = below_or_on_diag(
                    q_chunk_idx_start + q_chunk_idx,
                    query_chunk_size,
                    k_chunk_idx_start + k_chunk_idx,
                    key_chunk_size,
                    causal_block_size
                )
            return jax.lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (carry, None),
                carry,
                args
            )

        # Perform the attention on individual chunks
        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half, init=(numerator_chunk, denominator_chunk, max_score_chunk), xs=(k, v, jnp.arange(0, num_kv))
        )
        
        output_chunk = numerator_chunk / rearrange(denominator_chunk, 'b h q -> b q h')[..., None].astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

    # Perform the actual attention
    _, (_, numerator, denominator, max_score) = lax.scan(
        scan_attention, init=(), xs=(q, numerator, denominator, max_score, jnp.arange(0, num_q))
    )

    # Reshape back into original format:
    #   numerator is (batch, block_size, num_heads, dim_per_head)
    #   denominator is (batch, num_heads, block_size)
    #   max_score is (batch, num_heads, block_size)
    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, 'n b h c -> b h n c'), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score

def _blockwise_attention_bwd(q, k, v, g, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal_block_size, query_chunk_size, key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse, cache_idx):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry

    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, 'b h n c -> n b h c'), (denominator, max_score))

    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: jnp.moveaxis(x, 1, 0), (q, k, v))

    scale = jnp.sqrt(q.shape[-1])
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, segment_ids, deterministic,
        attn_dropout, attn_pdrop, causal_block_size, dtype)
    def scan_attention(carry, scan):
        dk, dv = carry
        q_chunk, dq_chunk, g_chunk, output_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]
        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk, precision=precision) / scale
            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk
            exp_weights = jnp.exp(attn_weights - max_score_chunk[..., None]) / denominator_chunk[..., None]

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)
            return dq_chunk, (dk_chunk, dv_chunk)

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            should_run = jnp.array(True)
            if causal_block_size is not None:
                should_run = below_or_on_diag(
                    q_chunk_idx_start + q_chunk_idx,
                    query_chunk_size,
                    k_chunk_idx_start + k_chunk_idx,
                    key_chunk_size,
                    causal_block_size
                )
            return lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (
                    carry, (
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                        jnp.zeros((batch, key_chunk_size, num_heads, dim_per_head), dtype=dk.dtype),
                    )
                ),
                carry,
                args
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half, init=dq_chunk, xs=(k, v, jnp.arange(0, num_kv))
        )
        return (dk + dk_part, dv + dv_part), dq_chunk
    (dk, dv), dq = lax.scan(scan_attention, init=(dk, dv), xs=(q, dq, g, output, denominator, max_score, jnp.arange(0, num_q)))

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv

def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, segment_ids, deterministic, attn_dropout, attn_pdrop, causal_block_size,
            dtype, query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, 0, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if segment_ids is not None:
        q_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, query_offset),
            slice_sizes=(segment_ids.shape[0], query_chunk_size)
        )
        k_segment_ids = lax.dynamic_slice(
            segment_ids,
            start_indices=(0, key_offset),
            slice_sizes=(segment_ids.shape[0], key_chunk_size)
        )
        segment_ids_mask = ~jnp.equal(q_segment_ids[:, :, None], k_segment_ids[:, None, :])
        segment_ids_mask = segment_ids_mask[:, None] # B1QK
        segment_ids_bias = segment_ids_mask * jnp.finfo(dtype).min
        chunk_bias = jnp.minimum(chunk_bias, segment_ids_bias)

    if causal_block_size is not None:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        query_idx += query_offset
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        key_idx += key_offset
        query_idx //= causal_block_size
        key_idx //= causal_block_size
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
        chunk_bias = jnp.minimum(chunk_bias, causal_mask_value.reshape(1, 1, *causal_mask_value.shape))

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)

def below_or_on_diag(r, r_blk_size, c, c_blk_size, causal_block_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    causal_block_size_q = max(causal_block_size, r_blk_size)
    causal_block_size_k = max(causal_block_size, c_blk_size)
    r = jax.lax.div(r, causal_block_size_q // r_blk_size)
    c = jax.lax.div(c, causal_block_size_k // c_blk_size)
    return ((r + 1) * causal_block_size_q - 1) > (c * causal_block_size_k)

def blockwise_feedforward(feedforward, inputs, chunk_size, static_argnums=(1,),
                          policy=jax.checkpoint_policies.nothing_saveable, pre_remat=True):
    if not pre_remat:
        remat_feedforward = partitioning.remat(feedforward, static_argnums=static_argnums, policy=policy)
    else:
        remat_feedforward = feedforward
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_feedforward(remat_feedforward, carry, hidden_states):
        outputs = remat_feedforward(hidden_states)
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, output = nn.scan(
        scan_feedforward,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_feedforward, None, inputs)
    output = rearrange(output, 'b c n d -> b (c n) d')
    return output
