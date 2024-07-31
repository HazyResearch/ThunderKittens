import torch
from causal_attention_cuda import causal_dot_product
torch.random.manual_seed(42)

b = 1
h = 32
l = 2048
d = 64
device = 'cuda'
eps = 1e-6

torch.set_default_dtype(torch.bfloat16)

q = torch.randn(b, h, l, d).to(device).contiguous() 
k = torch.randn(b, h, l, d).to(device).contiguous()
v = torch.randn(b, h, l, d).to(device).contiguous()

# For valid attention, the dot-prods b/t q and k should be positive
q, k, v = q.abs(), k.abs(), v.abs()

# Fast linear attention
z = 1 / (torch.einsum("nhli,nhli->nhl", q, k.cumsum(2)) + eps)
v_fast = causal_dot_product(q, k, v) * z[..., None]

# Slow linear attention
q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
print(q.shape, k.shape, v.shape)
v_slow = (q * (k * v).cumsum(dim=2)).sum(dim=-1) / ((q * k.cumsum(dim=2)).sum(dim=-1) + eps)

print((v_fast - v_slow).abs().max())  # 7.444406946888193e-05
print(v_fast.shape)
print(v_slow.shape)
# print(v_fast)
# print(v_slow)
