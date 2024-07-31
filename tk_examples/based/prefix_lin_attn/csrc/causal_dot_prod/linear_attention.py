import torch
from torch.utils.benchmark import Compare, Timer
import matplotlib.pyplot as plt
from causal_attention_cuda import causal_dot_product
import opt_einsum as oe

def linear_attention(q, k, v, eps=1e-6, qk_normalize=True, causal=False):
        
    # Expand dims for broadcasting to compute linear attention
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
    # breakpoint()
    
    if causal:
        a = (q * (k * v).cumsum(dim=2)).sum(dim=-1)
        if qk_normalize:
            a /= (q * k.cumsum(dim=2)).sum(dim=-1).clamp(eps)
    else:
        a = (q * (k * v).sum(dim=2, keepdim=True)).sum(dim=-1)
        if qk_normalize:
            a /= (q * k.sum(dim=2, keepdim=True)).sum(dim=-1).clamp(eps)
    return a
        
def fast_linear_attention(q, k, v, out, eps=1e-6, qk_normalize=True, causal=False):
    causal_dot_product(q, k, v, out)
    if qk_normalize:
        out /= oe.contract('n h l d, n h l d -> n h l', q, k.cumsum(2))[:, :, :, None]
    return out
  
def benchmark_memory(fn, *inputs, desc='', verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)
    if verbose:
        print(f'{desc} max memory: {mem}GB')
    torch.cuda.empty_cache()
    return mem

device = 'cuda'
b = 1
h = 12
d = 64
l = 1024       


seqlen_range = [128, 256, 512, 1024, 2048, 4096, 8192]  # , 16384, 32768, 65536]
results = []
mem_usage = []

for l in seqlen_range:
  q = torch.randn(b, h, l, d, device=device)
  k = torch.randn(b, h, l, d, device=device)
  v = torch.randn(b, h, l, d, device=device)
  label = 'Causal Linear Attention'
  sub_label = f'{l}'
  results.append(Timer(
      stmt='attn(q, k, v, qk_normalize=True, causal=True)',
      globals={'q': q, 'k': k, 'v': v, 'attn': linear_attention},
      label=label,
      sub_label=sub_label,
      description='Sequence Length vs Runtime',
  ).blocked_autorange(min_run_time=1))
  mem_usage.append(benchmark_memory(linear_attention, q, k, v, qk_normalize=True, causal=True))
compare = Compare(results)
compare.print()
print(mem_usage)

causal_linear_attn_run_times = []
causal_linear_attn_mem_usage = []
for result in results:
  causal_linear_attn_run_times.append(result.median * 1000)
  
for mem in mem_usage:
  causal_linear_attn_mem_usage.append(mem)



results = []
mem_usage = []
for l in seqlen_range:
  q = torch.randn(b, h, l, d, device=device)
  k = torch.randn(b, h, l, d, device=device) 
  v = torch.randn(b, h, l, d, device=device)
  label = 'Causal Scaled Dot Product Attention'
  sub_label = f'{l}'
  results.append(Timer(
      stmt='attn(q, k, v, attn_mask=None, is_causal=True)',
      globals={'q': q, 'k': k, 'v': v, 'attn': torch.nn.functional.scaled_dot_product_attention},
      label=label,
      sub_label=sub_label,
      description='Sequence Length vs Runtime',
  ).blocked_autorange(min_run_time=1))
  mem_usage.append(benchmark_memory(torch.nn.functional.scaled_dot_product_attention, q, k, v, attn_mask=None, is_causal=True))
compare = Compare(results)
compare.print()

causal_sdp_attn_run_times = []
causal_sdp_attn_mem_usage = []
for result in results:
  causal_sdp_attn_run_times.append(result.median * 1000)
  
for mem in mem_usage:
    causal_sdp_attn_mem_usage.append(mem)
  

results = []
mem_usage = []
for l in seqlen_range:
  q = torch.randn(b, h, l, d, device=device)
  k = torch.randn(b, h, l, d, device=device) 
  v = torch.randn(b, h, l, d, device=device)
  out = torch.zeros((b, h, l, d), device=device)
  label = 'Fast Causal Linear Attention'
  sub_label = f'{l}'
  results.append(Timer(
      stmt='attn(q, k, v, out)',
      globals={'q': q, 'k': k, 'v': v, 'attn': fast_linear_attention, 'out': out},
      label=label,
      sub_label=sub_label,
      description='Sequence Length vs Runtime',
  ).blocked_autorange(min_run_time=1))
  mem_usage.append(benchmark_memory(fast_linear_attention, q, k, v, out))
compare = Compare(results)
compare.print()

fast_causal_attn_run_times = []
fast_causal_attn_mem_usage = []
for result in results:
  fast_causal_attn_run_times.append(result.median * 1000)
  
for mem in mem_usage:
    fast_causal_attn_mem_usage.append(mem)
    
plt.plot(seqlen_range, causal_linear_attn_run_times, '-*', label='Causal Linear Attention')
plt.plot(seqlen_range, causal_sdp_attn_run_times, '-*', label='Causal Scaled Dot Product Attention')
plt.plot(seqlen_range, fast_causal_attn_run_times, '-*', label='Fast Causal Linear Attention')
plt.xlabel('Sequence Length')
plt.ylabel('Runtime (ms)')
plt.legend()
plt.grid()
plt.savefig('causal_runtime.png')


plt.figure()
plt.plot(seqlen_range, causal_linear_attn_mem_usage, '-*', label='Causal Linear Attention')
plt.plot(seqlen_range, causal_sdp_attn_mem_usage, '-*', label='Causal Scaled Dot Product Attention')
plt.plot(seqlen_range, fast_causal_attn_mem_usage, '-*', label='Fast Causal Linear Attention')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Usage (GB)')
plt.legend()
plt.grid()
plt.savefig('causal_mem_usage.png')


results = []
mem_usage = []
for l in seqlen_range:
  q = torch.randn(b, h, l, d, device=device)
  k = torch.randn(b, h, l, d, device=device)
  v = torch.randn(b, h, l, d, device=device)
  label = 'Bidirectional Linear Attention'
  sub_label = f'{l}'
  results.append(Timer(
      stmt='attn(q, k, v, qk_normalize=True, causal=False)',
      globals={'q': q, 'k': k, 'v': v, 'attn': linear_attention},
      label=label,
      sub_label=sub_label,
      description='Sequence Length vs Runtime',
  ).blocked_autorange(min_run_time=1))
  mem_usage.append(benchmark_memory(linear_attention, q, k, v, qk_normalize=True, causal=False))
compare = Compare(results)
compare.print()

bidirectional_linear_attn_run_times = []
for result in results:
  bidirectional_linear_attn_run_times.append(result.median * 1000)

bidirectional_linear_mem_usage = []
for mem in mem_usage:
    bidirectional_linear_mem_usage.append(mem)
    
    
results = []
mem_usage = []
for l in seqlen_range:
  q = torch.randn(b, h, l, d, device=device)
  k = torch.randn(b, h, l, d, device=device) 
  v = torch.randn(b, h, l, d, device=device)
  label = 'Bidirectional Scaled Dot Product Attention'
  sub_label = f'{l}'
  results.append(Timer(
      stmt='attn(q, k, v, attn_mask=None, is_causal=False)',
      globals={'q': q, 'k': k, 'v': v, 'attn': torch.nn.functional.scaled_dot_product_attention},
      label=label,
      sub_label=sub_label,
      description='Sequence Length vs Runtime',
  ).blocked_autorange(min_run_time=1))
  mem_usage.append(benchmark_memory(torch.nn.functional.scaled_dot_product_attention, q, k, v, attn_mask=None, is_causal=False))
compare = Compare(results)
compare.print()

bidirectional_sdp_attn_run_times = []
for result in results:
  bidirectional_sdp_attn_run_times.append(result.median * 1000)
  
bidirectional_sdp_mem_usage = []
for mem in mem_usage:
    bidirectional_sdp_mem_usage.append(mem)
    
plt.figure()
plt.plot(seqlen_range, bidirectional_linear_attn_run_times, '-*', label='Bidirectional Linear Attention')
plt.plot(seqlen_range, bidirectional_sdp_attn_run_times, '-*', label='Bidirectional Scaled Dot Product Attention')
plt.xlabel('Sequence Length')
plt.ylabel('Runtime (ms)')
plt.legend()
plt.grid(True)
plt.savefig('bidirectional_runtime.png')


plt.figure()
plt.plot(seqlen_range, bidirectional_linear_mem_usage, '-*', label='Bidirectional Linear Attention')
plt.plot(seqlen_range, bidirectional_sdp_mem_usage, '-*', label='Bidirectional Scaled Dot Product Attention')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Usage (GB)')
plt.legend()
plt.grid(True)
plt.savefig('bidirectional_mem_usage.png')

# repeats = 11

# import time

# linear_attention(q, k, v, qk_normalize=False, causal=True)
# torch.cuda.synchronize()
# start = time.time()
# for _ in range(repeats):
#     linear_attention(q, k, v, qk_normalize=False, causal=True)
# torch.cuda.synchronize()
# end = time.time()

# print("Linear Attention Time taken: ", (end - start) / repeats)


# torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
# torch.cuda.synchronize()
# start = time.time()
# for _ in range(repeats):
#     torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
# torch.cuda.synchronize()
# end = time.time()

# print("Scaled Dot Product Attention Time taken: ", (end - start) / repeats)

# d = 768
# q = torch.randn(b, h, l, d, device=device)
# k = torch.randn(b, h, l, d, device=device)
# v = torch.randn(b, h, l, d, device=device)


# repeats = 5
# linear_attention(q, k, v, qk_normalize=True, causal=True)
# torch.cuda.synchronize()
# start = time.time()
# for _ in range(repeats):
#     linear_attention(q, k, v, qk_normalize=True, causal=True)
# torch.cuda.synchronize()
# end = time.time()
# print("Linear Attention Time taken: ", (end - start) / repeats)


# torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
# torch.cuda.synchronize()
# start = time.time()
# for _ in range(repeats):
#     torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
# torch.cuda.synchronize()
# end = time.time()

# print("Scaled Dot Product Attention Time taken: ", (end - start) / repeats)

# # print(linear_attention(q, k, v, qk_normalize=True, causal=True).shape)




# #Linear Attention VS sequence length
# b = 4
# h = 12
# l = 1024
# d = 64        
# q = torch.randn(b, h, l, d)
# k = torch.randn(b, h, l, d)
# v = torch.randn(b, h, l, d)


# # from torch.utils.benchmark import Compare, Timer

# # results = []
# # seqlen_range = [128, 256, 512, 1024, 2048, 4096, 8192]
# # for L in seqlen_range:
# #   x = torch.randnn(b, L, D, device=device)
# #   attn = CausalSelfAttention(D, H, L)
# #   attn.to(device)
# #   label = 'Multi-headed Causal Self Attention'
# #   sub_label = f'{L}'
# #   results.append(Timer(
# #       stmt='attn(x)',
# #       globals={'x': x, 'attn': attn},
# #       label=label,
# #       sub_label=sub_label,
# #       description='Sequence Length vs Runtime',
# #   ).blocked_autorange(min_run_time=1))
# # compare = Compare(results)
# # compare.print()

# # medians = []
# # for result in results:
# #   medians.append(result.median * 1000)