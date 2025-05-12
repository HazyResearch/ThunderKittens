import torch
from torch import Tensor

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

# From transformers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# A little modification on the inputs
def matvec_rope(
    rms_rope_intermediates: Tensor,
    qkv_weights: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    pos_id: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int
):
    batch_size = rms_rope_intermediates.shape[0]

    matmul_output = torch.matmul(
        rms_rope_intermediates, # (B, D)
        qkv_weights.T, # (D, (H_q + H_k + H_v) * D_h)
    ) # (B, (H_q + H_k + H_v) * D_h)

    k_start = num_attention_heads * head_dim
    v_start = k_start + num_kv_heads * head_dim

    q = matmul_output[:, :k_start] # (B, H_q * D_h)
    k = matmul_output[:, k_start:v_start] # (B, H_k * D_h)
    v = matmul_output[:, v_start:] # (B, H_v * D_h)

    q_arr = torch.reshape(q, (batch_size, num_attention_heads, head_dim)) # (B, H_q, D_h)
    k_arr = torch.reshape(k, (batch_size, num_kv_heads, head_dim)) # (B, H_k, D_h)

    q_with_rope = q_arr * rope_cos[pos_id+1][torch.newaxis, torch.newaxis, :] + rotate_half(q_arr) * rope_sin[pos_id+1][torch.newaxis, torch.newaxis, :]
    k_with_rope = k_arr * rope_cos[pos_id+1][torch.newaxis, torch.newaxis, :] + rotate_half(k_arr) * rope_sin[pos_id+1][torch.newaxis, torch.newaxis, :]

    q_with_rope = q_with_rope.view(batch_size, -1) # (B, H_q * D_h)
    k_with_rope = k_with_rope.view(batch_size, -1).reshape(batch_size, num_kv_heads, head_dim) # (B, H_k, D_h)
    v = v.reshape(batch_size, num_kv_heads, head_dim) # (B, H_v, D_h)

    return q_with_rope, k_with_rope, v
