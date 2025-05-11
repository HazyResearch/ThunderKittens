# From Jordan's Python KVM code

from einops import einsum, rearrange
import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end

# For compatibility with our kernel
# Inefficient but quick impl
def uninterleave(x: Tensor):
    assert(x.shape[-1] % 128 == 0)
    y = torch.zeros_like(x)
    for i in range(0, x.shape[-1], 128):
        y[..., i:i+64] = x[..., i:i+128:2]
        y[..., i+64:i+128] = x[..., i+1:i+128:2]
    return y
def interleave(x: Tensor):
    assert(x.shape[-1] % 128 == 0)
    y = torch.zeros_like(x)
    for i in range(0, x.shape[-1], 128):
        y[..., i:i+128:2] = x[..., i:i+64]
        y[..., i+1:i+128:2] = x[..., i+64:i+128]
    return y

# A little modification on the inputs
def layer_norm_matvec_rope_append(
    rms_rope_intermediates: Tensor,
    qkv_weights: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    q_post_rope: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    layer_idx: int,
    pos_id: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qkv_block_size: int,
    output_block_idx: int,
):
    batch_size = rms_rope_intermediates.shape[0]

    matmul_output = einsum(
        rms_rope_intermediates,
        qkv_weights[layer_idx],
        'ik,jk->ij'
    )
    matmul_output = uninterleave(matmul_output)

    k_start = num_attention_heads * head_dim
    v_start = k_start + num_kv_heads * head_dim

    q = matmul_output[:, :k_start]
    k = matmul_output[:, k_start:v_start]
    v = matmul_output[:, v_start:]

    q_arr = rearrange(q, "b (h d) -> b h d", h=num_attention_heads)
    k_arr = rearrange(k, "b (h d) -> b h d", h=num_kv_heads)

    q_with_rope, k_with_rope = apply_rotary_pos_emb(
        q=q_arr,
        k=k_arr,
        cos=uninterleave(rope_cos[pos_id].unsqueeze(0)),
        sin=uninterleave(rope_sin[pos_id].unsqueeze(0)),
        unsqueeze_dim=0,
    )

    q_with_rope = interleave(q_with_rope.view(batch_size, -1)).reshape(batch_size, num_attention_heads, head_dim)
    k_with_rope = interleave(k_with_rope.view(batch_size, -1)).reshape(batch_size, num_kv_heads, head_dim)
    v = interleave(v.view(batch_size, -1)).reshape(batch_size, num_kv_heads, head_dim)

    start, end = get_start_end(qkv_block_size, output_block_idx)
    if start < k_start:
        q_post_rope[:, start:end] = q_with_rope[:, start:end]
    elif start < v_start:
        start_in_k = start - k_start
        end_in_k = end - k_start
        k_cache[layer_idx, pos_id].view(batch_size, -1)[:, start_in_k:end_in_k] = (
            k_with_rope.view(batch_size, -1)[:, start_in_k:end_in_k]
        )
    else:
        start_in_v = start - v_start
        end_in_v = end - v_start
        v_cache[layer_idx, pos_id].view(batch_size, -1)[:, start_in_v:end_in_v] = v.view(batch_size, -1)[
            start_in_v:end_in_v
        ]
