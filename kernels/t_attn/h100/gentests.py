import torch
from tqdm import trange
from einops import rearrange, repeat, einsum
import numpy as np
import sys
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1

N = int(sys.argv[1])
D = int(sys.argv[2])

torch.random.manual_seed(42)

q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
b = (torch.randn((B, H, N, N), dtype=torch.bfloat16, device='cuda'))

# implementation of Attend forward used in Triangle Attention (lucidrains repo for AF3 PyTorch)
# Triangle Attention: https://github.com/lucidrains/alphafold3-pytorch/blob/e4202e312c874954fb0944e3393a07b4fbe86d4e/alphafold3_pytorch/alphafold3.py#L847-L848
# Attention: https://github.com/lucidrains/alphafold3-pytorch/blob/e4202e312c874954fb0944e3393a07b4fbe86d4e/alphafold3_pytorch/attention.py#L166
# Attend: https://github.com/lucidrains/alphafold3-pytorch/blob/e4202e312c874954fb0944e3393a07b4fbe86d4e/alphafold3_pytorch/attention.py#L277

def forward(q, k, v, attn_bias):

    dtype = q.dtype

    # no local attention
    # if self.is_local_attn:
    #     return self.local_attn(q, k, v, mask = mask, windowed_mask = windowed_mask, attn_bias = attn_bias, memory_kv = memory_kv)

    # default attention
    scale = math.sqrt(q.shape[-1])
    q = q * scale

    # similarity
    sim = einsum(q, k, "b h i d, b h j d -> b h i j")

    # attn bias
    sim = sim + attn_bias

    # no masking
    # if exists(mask):
    #     sim = einx.where(
    #         'b j, b h i j, -> b h i j',
    #         mask, sim, max_neg_value(sim)
    #     )

    # attention cast float32 - in case there are instabilities with float16
    softmax_kwargs = dict()

    # attention
    attn = sim.softmax(dim = -1, **softmax_kwargs)
    attn = attn.to(dtype)

    # aggregate values
    out = einsum(attn, v, "b h i j, b h j d -> b h i d")

    return out

o = forward(q, k, v, b)

print("--------------------------------------")
print("Q shape: ",      q.shape)
print("K shape: ",      k.shape)
print("V shape: ",      v.shape)
print("B shape: ",      b.shape)
print("O shape: ",      o.shape)
print("--------------------------------------")

# print out avg magnitude of output tensor
print(f'Average magnitude of OUTPUT tensor: {o.abs().mean()}')
print(f'1/100 magnitude of OUTPUT tensor:   {o.abs().mean()/100}')
print("--------------------------------------")

filename = f'randn_{N}N_{D}D'

filename += '.txt'

with open(filename, 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    bf = b.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3]):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(k.shape[0] * k.shape[1] * k.shape[2] * k.shape[3]):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(v.shape[0] * v.shape[1] * v.shape[2] * v.shape[3]):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(b.shape[0] * b.shape[1] * b.shape[2] * b.shape[3]):
        f.write(repr(bf[i]))
        f.write(' ')
    for i in trange(o.shape[0] * o.shape[1] * o.shape[2] * o.shape[3]):
        f.write(repr(of[i]))
        f.write(' ')