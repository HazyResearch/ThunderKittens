import torch
import b200_attn

def ref_attn(q, k, v):
    B, H, N, D = q.shape
    a = torch.matmul(q, k.transpose(-2, -1))
    a = a / (D ** 0.5)
    a = torch.softmax(a, dim=-1)
    o = torch.matmul(a, v)
    return o

B = 1
H = 1
N = 8192
D = 128

q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
l = torch.empty((B, H, 1, N), dtype=torch.float, device='cuda')
o = torch.empty((B, H, N, D), dtype=torch.bfloat16, device='cuda')

b200_attn.fwd_attend_ker_128_noncausal(q, k, v, l, o)

breakpoint()
print(o.abs().max())

ref = ref_attn(q, k, v)
print(ref.abs().max())

print((o - ref).abs().max())
