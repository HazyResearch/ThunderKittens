import torch
import b200_attn
import time
print('Finished importing')

torch.manual_seed(42)

def ref_attn(q, k, v):
    B, H, N, D = q.shape
    a = torch.matmul(q, k.transpose(-2, -1))
    a = a / (D ** 0.5)
    a = torch.softmax(a, dim=-1)
    o = torch.matmul(a, v)
    return o

B = 1
H = 1
N = 1536
D = 128

print('Starting making tensors')
q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
# q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
# k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
# v = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
l = torch.zeros((B, H, 1, N), dtype=torch.float, device='cuda')
o = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')

stream = torch.cuda.current_stream()

# #warmup
# print('Starting warmup')
b200_attn.fwd_attend_ker_128_noncausal(q, k, v, l, o)
# torch.cuda.synchronize()
# print('Finished warmup')
# t0 = time.time()
# ITERS = 5
# for _ in range(ITERS):
#     b200_attn.fwd_attend_ker_128_noncausal(q, k, v, l, o)
# torch.cuda.synchronize()
# t1 = time.time()
# flops = 4 * B * H * N * N * D
# avg_us = (t1 - t0)/ITERS*1e6
# print(f'Time taken: {avg_us} us/iter')
# print(f'Total FLOPs: {flops}')
# print(f'TFLOPS: {(flops / (avg_us*1e-6))*10**-12} TFLOPS')

print(o.abs().max())

q.requires_grad = True
k.requires_grad = True
v.requires_grad = True
ref = ref_attn(q, k, v)
print(ref.abs().max())

print((o - ref).abs().max())
print((o - ref).abs().mean())

print('Starting backward')

# o_grad = torch.randn_like(ref)
o_grad = torch.ones_like(ref)
ref.backward(o_grad)

q_grad_ref = q.grad
k_grad_ref = k.grad
v_grad_ref = v.grad

def custom_backward(q, k, v, l, o, o_grad):
    # Create gradient tensors
    qg = torch.zeros_like(q, dtype=torch.float, device='cuda')
    kg = torch.zeros_like(k, dtype=torch.float, device='cuda')
    vg = torch.zeros_like(v, dtype=torch.float, device='cuda')

    print(qg.shape, kg.shape, vg.shape)
    
    # Create d vector tensor
    d_vec = torch.empty((B, H, 1, N), dtype=torch.float, device='cuda')
    b200_attn.bwd_attend_prep_ker_128(o_grad, o, d_vec)
    print(d_vec)
    b200_attn.bwd_attend_ker_128_noncausal(q, k, v, o_grad, qg, kg, vg, l, d_vec, q.shape[-2], 1)
    torch.cuda.synchronize()

    return qg, kg, vg

qg, kg, vg = custom_backward(q, k, v, l, o, o_grad)

def compare_grads(ref_grad, custom_grad, name):
    max_diff = (ref_grad - custom_grad).abs().max()
    mean_diff = (ref_grad - custom_grad).abs().mean()
    print(f'{name} Ref grad mean: {ref_grad.abs().mean()}, max: {ref_grad.abs().max()}')
    print(f'{name} Custom grad mean: {custom_grad.abs().mean()}, max: {custom_grad.abs().max()}')
    print(f'{name} Mean diff: {mean_diff}, Max diff: {max_diff}')

compare_grads(q_grad_ref, qg, 'Q')
compare_grads(k_grad_ref, kg, 'K')
compare_grads(v_grad_ref, vg, 'V')

breakpoint()
