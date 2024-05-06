import torch
import numpy as np
import sys
import math
from einops import rearrange, repeat
from tqdm import trange

# Configuration
B, H = 1, 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
D = int(sys.argv[3]) if len(sys.argv) > 3 else 128
softmax_scale = 1 / math.sqrt(D)

# Forward function
def forward(Q, K, V):
    attention_logits = torch.einsum("bhnd,bhmd->bhnm", Q, K) * softmax_scale
    attention = torch.softmax(attention_logits, dim=-1)
    output = torch.einsum("bhnm,bhmd->bhnd", attention, V)
    return output

# Backward function
def backward(Q, K, V, grad_output):
    attention_logits = torch.einsum("bhnd,bhmd->bhnm", Q, K) * softmax_scale
    attention = torch.softmax(attention_logits, dim=-1)
    
    grad_V = torch.einsum("bhnm,bhnd->bhmd", attention, grad_output)
    dL_dA = torch.einsum("bhnd,bhmd->bhnm", grad_output, V)
    dL_dAttention = attention * (dL_dA - (dL_dA * attention).sum(dim=-1, keepdim=True))
    
    grad_Q = torch.einsum("bhnm,bhmd->bhnd", dL_dAttention * softmax_scale, K)
    grad_K = torch.einsum("bhnm,bhnd->bhmd", dL_dAttention * softmax_scale, Q)
    
    return grad_Q, grad_K, grad_V

# Test configurations
TESTNAME = sys.argv[1]
if TESTNAME == 'ones':
    q = k = v = grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q = k = v = grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    eye_template = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1) * 10
    q = k = v = grad_output = eye_template.to(dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'v_orientation':
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda') / D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
    q = k = grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
else:
    print('Invalid test name')
    sys.exit(0)

l_vec = torch.einsum("bhnd,bhmd->bhnm", q.clone(), k.clone()) * softmax_scale
max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec = l_vec.sum(dim=-1, keepdim=True)

# Actual forward and backward
o = forward(q, k, v)
q_grad, k_grad, v_grad = backward(q, k, v, grad_output)

# Intermediate values
l_vec = max_vec + torch.log(l_vec)

d_vec = torch.mul(grad_output, o)
d_vec = d_vec.sum(dim=-1, keepdim=True)

fn = f'{TESTNAME}_{N}_{D}.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    grad_outputf = grad_output.to(torch.float32).flatten().cpu().numpy()
    
    # intermediate
    l_vecf = l_vec.to(torch.float32).flatten().cpu().numpy()
    d_vecf = d_vec.to(torch.float32).flatten().cpu().numpy()
    
    # outputs
    q_grad = q_grad.to(torch.float32).flatten().cpu().numpy()
    k_grad = k_grad.to(torch.float32).flatten().cpu().numpy()
    v_grad = v_grad.to(torch.float32).flatten().cpu().numpy()
    
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*N):
        f.write(repr(l_vecf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(grad_outputf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(q_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(k_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(v_grad[i]))
        f.write(' ')
        
print(f'Run the harness like `./attn_bwd {fn}`')
    
        
############################################################
## BENCHMARKING
############################################################
N = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 64
H = 16
B = 16

torch.use_deterministic_algorithms(False)

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

print("-" * 60)
print(f'Timing forward pass for for B={B}, H={H}, N={N}, D={D}')
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, 
    enable_math=False, 
    enable_mem_efficient=False
):
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()

    # warmup
    for _ in range(10):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

    # Time the forward pass

    for i in range(30):
        start_events[i].record()
        torch.cuda.synchronize()
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end_events[i].record()
    
torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
time_us = np.mean(times) * 1000
print(f'Average time for forward pass in us: {time_us:.2f}')
print(f'Average efficiency for forward pass in TFLOPS: {efficiency(flops(B, N, D, H, False, "fwd"), time_us):.2f}')
print("-" * 60)
print(f'Timing backwards pass for B={B}, H={H}, N={N}, D={D}')

with torch.backends.cuda.sdp_kernel(
    enable_flash=True, 
    enable_math=False, 
    enable_mem_efficient=False
):
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    grad_output = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda')
        
    # warmup
    for _ in range(10):
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v).requires_grad_()
        
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        o.backward(grad_output, retain_graph=True)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

    # Time the backward pass

    for i in range(30):
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None

        o = torch.nn.functional.scaled_dot_product_attention(q, k, v).requires_grad_()
        
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        start_events[i].record()
        torch.cuda.synchronize()
        o.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        end_events[i].record()
    
torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

time_us = np.mean(times) * 1000

print(f'Average time for backward pass in us: {time_us:.2f}')
print(f'Average efficiency for backward pass in TFLOPS: {efficiency(flops(B, N, D, H, False, "bwd"), time_us):.2f}')
print("-" * 60)
print(f'Timing forward + backward pass for B={B}, H={H}, N={N}, D={D}')

with torch.backends.cuda.sdp_kernel(
    enable_flash=True, 
    enable_math=False, 
    enable_mem_efficient=False
):
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda').requires_grad_()
    grad_output = torch.randn((B, H, N, D), dtype=torch.float16, device='cuda')
        
    # warmup
    for _ in range(10):
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v).requires_grad_()
        
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
    
        o.backward(grad_output, retain_graph=True)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

    # Time the forward and backward pass

    for i in range(30):
        start_events[i].record()
        torch.cuda.synchronize()
        
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v).requires_grad_()
        
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        o.grad = None
        
        o.backward(grad_output, retain_graph=True)
        
        torch.cuda.synchronize()
        end_events[i].record()
    
torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
time_us = np.mean(times) * 1000
print(f'Average time for forward + backward pass in us: {time_us:.2f}')
print(f'Average efficiency for forward + backward pass in TFLOPS: {efficiency(flops(B, N, D, H, False, "fwd_bwd"), time_us):.2f}')
print("-" * 60)
