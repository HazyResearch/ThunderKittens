import torch
import b200_attn
import time
print('Finished importing')

def ref_attn(q, k, v):
    B, H, N, D = q.shape
    a = torch.matmul(q, k.transpose(-2, -1))
    a = a / (D ** 0.5)
    a = torch.softmax(a, dim=-1)
    o = torch.matmul(a, v)
    return o

B = 16
H = 16
N = 8192
D = 128

print('Starting making tensors')
q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
l = torch.empty((B, H, 1, N), dtype=torch.float, device='cuda')
o = torch.empty((B, H, N, D), dtype=torch.bfloat16, device='cuda')

stream = torch.cuda.current_stream()

#warmup
print('Starting warmup')
b200_attn.fwd_attend_ker_128_noncausal(q, k, v, l, o)
torch.cuda.synchronize()
print('Finished warmup')
t0 = time.time()
ITERS = 5
for _ in range(ITERS):
    b200_attn.fwd_attend_ker_128_noncausal(q, k, v, l, o)
torch.cuda.synchronize()
t1 = time.time()
flops = 4 * B * H * N * N * D
avg_us = (t1 - t0)/ITERS*1e6
print(f'Time taken: {avg_us} us/iter')
print(f'Total FLOPs: {flops}')
print(f'TFLOPS: {(flops / (avg_us*1e-6))*10**-12} TFLOPS')

print(o.abs().max())

ref = ref_attn(q, k, v)
print(ref.abs().max())

print((o - ref).abs().max())
print((o - ref).abs().mean())
