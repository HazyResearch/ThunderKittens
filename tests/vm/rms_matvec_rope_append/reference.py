# From Jordan's Python KVM code

from einops import einsum, rearrange
import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end

# For compatibility with our kernel
# Inefficient but quick impl
def uninterleave(x: Tensor):
    assert(len(x.shape) == 1 and x.shape[0] % 64 == 0)
    y = torch.zeros_like(x)
    for i in range(0, x.shape[0], 64):
        y[i:i+32] = x[i:i+64:2]
        y[i+32:i+64] = x[i+1:i+64:2]
    return y
def interleave(x: Tensor):
    assert(len(x.shape) == 1 and x.shape[0] % 64 == 0)
    y = torch.zeros_like(x)
    for i in range(0, x.shape[0], 64):
        y[i:i+64:2] = x[i:i+32]
        y[i+1:i+64:2] = x[i+32:i+64]
    return y

# A little modification on the inputs
def layer_norm_matvec_rope_append(
    # Globals
    hidden_states: Tensor,
    attn_ln_weight: Tensor,
    qkv_proj: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    post_ln_rope_q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    pos_id: int,

    # Globals, but fixed
    qkv_block_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ln_eps: float,

    # Instruction
    layer_idx: int,
    output_block_idx: int
):
    post_ln = rms_norm(
        inp=hidden_states,
        weight=attn_ln_weight[layer_idx],
        eps=ln_eps,
    )

    matmul_output = einsum(
        qkv_proj[layer_idx],
        post_ln,
        "o i, i -> o",
    )
    matmul_output = uninterleave(matmul_output)

    k_start = num_attention_heads * head_dim
    v_start = k_start + num_kv_heads * head_dim

    q = matmul_output[:k_start]
    k = matmul_output[k_start:v_start]
    v = matmul_output[v_start:]

    q_arr = rearrange(q, "(h d) -> h d", h=num_attention_heads)
    k_arr = rearrange(k, "(h d) -> h d", h=num_kv_heads)

    pos_id = pos_id
    q_with_rope, k_with_rope = apply_rotary_pos_emb(
        q=q_arr,
        k=k_arr,
        cos=uninterleave(rope_cos[pos_id]),
        sin=uninterleave(rope_sin[pos_id]),
        unsqueeze_dim=0,
    )

    q_with_rope = interleave(q_with_rope.view(-1)).reshape(32, 64)
    k_with_rope = interleave(k_with_rope.view(-1)).reshape(8, 64)
    v = interleave(v.view(-1))

    start, end = get_start_end(qkv_block_size, output_block_idx)
    if start < k_start:
        post_ln_rope_q[start:end] = q_with_rope.view(-1)[start:end]
    elif start < v_start:
        start_in_k = start - k_start
        end_in_k = end - k_start
        k_cache[layer_idx, pos_id].view(-1)[start_in_k:end_in_k] = (
            k_with_rope.view(-1)[start_in_k:end_in_k]
        )
    else:
        start_in_v = start - v_start
        end_in_v = end - v_start
        v_cache[layer_idx, pos_id].view(-1)[start_in_v:end_in_v] = v.view(-1)[
            start_in_v:end_in_v
        ]
