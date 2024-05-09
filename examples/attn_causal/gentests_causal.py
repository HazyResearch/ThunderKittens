import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = int(sys.argv[1])
D = int(sys.argv[2])

TESTNAME = sys.argv[3] if len(sys.argv) > 3 else 'randn'

if TESTNAME == 'ones':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
else:
    print('Invalid test name')
    sys.exit(0)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

with open(f'{TESTNAME}_causal_{N}N_{D}D.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
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

print(f'Run the harness like `cat {TESTNAME}.txt | ./harness`')

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

B = 32
H = 16
N = int(sys.argv[1])
D = int(sys.argv[2])

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
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

    # Time the forward pass

    for i in range(30):
        start_events[i].record()
        torch.cuda.synchronize()
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        end_events[i].record()
    
torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
time_us = np.mean(times) * 1000
print(f'Average time for forward pass in us: {time_us:.2f}')
print(f'Average efficiency for forward pass in TFLOPS: {efficiency(flops(B, N, D, H, True, "fwd"), time_us):.2f}')

print("-" * 60)