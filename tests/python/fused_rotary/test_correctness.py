import torch
from einops import rearrange, repeat
from baselines.rotary import RotaryEmbedding
import thunderkittens as tk


head_dim = 64
rotary_emb_fraction = 1
rotary_emb_dim = head_dim * rotary_emb_fraction
rotary_emb_base = 10000
rotary_emb_scale_base=None
rotary_emb_interleaved=False
device='cuda'

rotary_emb = RotaryEmbedding(
    rotary_emb_dim,
    base=rotary_emb_base,
    scale_base=rotary_emb_scale_base,
    interleaved=rotary_emb_interleaved,
    device=device,
)

dt = torch.bfloat16

rotary_max_seqlen = None
b, h, n, d = 2, int(1024/head_dim), 1024, head_dim
qkv = torch.randn((b, n, 3, h, d), dtype=dt, device=device) / (d)
qkv_clone = qkv.clone().detach()
qkv_clone_manual = qkv.clone().detach()
qkv_clone_tk = torch.ones((b, h, n, d), dtype=dt, device=device)  * qkv[:,:,0].transpose(1,2).detach()


# Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L957
qkv_emb_flash = rotary_emb(
    qkv.to(dt), seqlen_offset=0, max_seqlen=rotary_max_seqlen
)

def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    o_dt = x.dtype
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim].to(dt) * cos.to(dt) + rotate_half(x[..., :ro_dim].to(dt), interleaved).to(dt) * sin.to(dt), x[..., ro_dim:].to(dt)],
        dim=-1,
    ).to(o_dt)


def apply_rotary_emb_manual(x, cos_in, sin_in, interleaved=False):

    o_dt = x.dtype
    ro_dim = cos_in.shape[-1] * 2      # rotary_emb_dim * 2
    assert ro_dim <= x.shape[-1]    # assert

    x_ro_dim = x[..., :ro_dim].to(dt)
    x_ro_dim_end = x[..., ro_dim:].to(dt)
    print(f"{x_ro_dim.shape=}, {x_ro_dim_end.shape=}")

    x1, x2 = x_ro_dim.chunk(2, dim=-1)
    rotated_x = torch.cat((-x2, x1), dim=-1)
    
    cos = repeat(cos_in, "... d -> ... 1 (2 d)" ).to(dt)  # double the dimension
    sin = repeat(sin_in, "... d -> ... 1 (2 d)" ).to(dt)  # double the dimension
    return torch.cat([x_ro_dim * cos + rotated_x * sin, x_ro_dim_end], dim=-1).to(o_dt)


def _update_cos_sin_cache(seqlen, base, dim, device=None):
    t = torch.arange(seqlen, device=device, dtype=dt) # We want fp32 here

    # We want to recompute self.inv_freq if it was not loaded in fp32
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dt) / dim))

    # Don't do einsum, it converts fp32 to fp16 under AMPl freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    freqs = torch.outer(t, inv_freq).to(dtype=dt)
    _cos_cached = torch.cos(freqs).to(dt)
    _sin_cached = torch.sin(freqs).to(dt)
    return _cos_cached, _sin_cached


def apply_rotary_emb_tk(x, cos_in, sin_in, interleaved=False):
    b, n, h, dh = x.shape 
    out = torch.zeros_like(x)
    tk.fused_rotary(
        x.contiguous(), 
        cos_in.contiguous(), 
        sin_in.contiguous(), 
        out.contiguous()
    )
    out = rearrange(out, 'b h n d -> b n h d')
    return out 

# difference
sin_ref = rotary_emb._sin_cached
cos_ref = rotary_emb._cos_cached
cos, sin = _update_cos_sin_cache(n, rotary_emb_base, rotary_emb_dim, device=device)
diff_cos = torch.norm(cos_ref - cos).item()
diff_sin = torch.norm(sin_ref - sin).item()
print(f"diff cos = {diff_cos}, diff sin = {diff_sin}")

q_emb_torch = apply_rotary_emb_torch(qkv_clone[:,:,0], cos_ref, sin_ref, interleaved=rotary_emb_interleaved)
k_emb_torch = apply_rotary_emb_torch(qkv_clone[:,:,1], cos_ref, sin_ref, interleaved=rotary_emb_interleaved)
q_emb_manual = apply_rotary_emb_manual(qkv_clone_manual[:,:,0], cos_ref, sin_ref, interleaved=rotary_emb_interleaved)
if dt == torch.bfloat16:
    q_emb_tk    = apply_rotary_emb_tk(qkv_clone_tk, cos_ref, sin_ref, interleaved=rotary_emb_interleaved)
    k_emb_tk    = apply_rotary_emb_tk(qkv_clone_tk, cos_ref, sin_ref, interleaved=rotary_emb_interleaved)

q_diff = torch.norm(qkv_emb_flash[:,:,0,:,:] - q_emb_torch).item()
k_diff = torch.norm(qkv_emb_flash[:,:,1,:,:] - k_emb_torch).item()
print(f"q_diff flash - torch: {q_diff}")
print(f"k_diff flash - torch: {k_diff}")

q_diff = torch.norm(qkv_emb_flash[:,:,0,:,:] - q_emb_manual).item()
print(f"q_diff torch - manual: {q_diff}")

if dt == torch.bfloat16:
    q_diff = torch.norm(qkv_emb_flash[:,:,0,:,:] - q_emb_tk).item()
    print(f"q_diff torch - tk: {q_diff}")

print("Sample of flash output:")
print(qkv_emb_flash[1,4,0,8,:16])
print("Sample of tk output:")
print(q_emb_tk[1,4,8,:16])
