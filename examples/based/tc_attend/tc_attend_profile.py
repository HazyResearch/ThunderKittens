import torch
debug=False
from test_build_utils import __eq
import tc_attend as mod
dtypes = [torch.bfloat16]

import torch
from flash_attn import flash_attn_qkvpacked_func
import time
from statistics import median
from collections import defaultdict
import matplotlib.pyplot as plt


WINDOW_SIZE = 128


def run_fa2_test(dt, s, window_size=64, qw=None, kw=None, vw=None):

    b, h, n, d = qw.shape

    try:
        # make the shape [B, N, 3, H, D]
        qkv = torch.cat([qw, kw, vw], dim=-2).view(b, h, n, 3, d).permute(0, 2, 3, 1, 4)

        if window_size is not None:

            # benchmark
            torch.cuda.synchronize()
            t0 = time.time()
            result = flash_attn_qkvpacked_func(
                qkv,
                0.0, # dropout
                softmax_scale=0.5,
                causal=True, 
                window_size=(window_size,0),
            )
            torch.cuda.synchronize()
            t1 = time.time()
        else:

            # benchmark
            torch.cuda.synchronize()
            t0 = time.time()
            result = flash_attn_qkvpacked_func(
                qkv,
                0.0, # dropout
                softmax_scale=0.5,
                causal=True, 
            )
            torch.cuda.synchronize()
            t1 = time.time()
        tot = t1-t0
    except:
        tot = -1
        result = None

    return tot, result


def run_torch_test(dt, s, window_size=64, qw=None, kw=None, vw=None):
    
    B, H, N, D = qw.shape

    try:

        # benchmark
        torch.cuda.synchronize()
        t0 = time.time()

        # Start
        att = torch.einsum('bhtd,bhsd->bhts', qw.to(torch.float32), kw.to(torch.float32))
        att /= D**.5
        att = torch.exp(att)
        for t in range(N):
            att[:,:,t,t+1:] = 0 # make causal
            if t > window_size:
                att[:,:,t,:16*((t-window_size)//16)] = 0 # make sliding
        att /= att.sum(dim=-1, keepdims=True) # normalize softmax
        o = torch.einsum('bhts,bhsd->bhtd', att.to(torch.bfloat16), vw)
        # End
        
        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0

    except:
        tot = -1
        o = None

    return tot, o


def run_based_test(dt, s, window_size=64, qw=None, kw=None, vw=None):
    b, h, n, d = s
    o     = torch.zeros((b,h,n,d), dtype=dt, device='cuda')

    try:
        # benchmark kernel
        torch.cuda.synchronize()
        t0 = time.time()

        mod.sliding_window_tk(qw, kw, vw, o)

        torch.cuda.synchronize()
        t1 = time.time()
        tot = t1-t0
    except:
        tot = -1
        o = None

    return tot, o


def sliding_window_benchmark_batch(dt):
    num_iters = 4
    
    methods = {
        'Flash Attention (Sliding Window)': run_fa2_test, 
        'Flash Attention (no Sliding Window)': run_fa2_test, 
        'Pure PyTorch': run_torch_test, 
        'Based Sliding Window Kernel': run_based_test
    }
    method2timing = defaultdict(dict)
    for i, B in enumerate([1, 2, 8, 16, 32, 64, 128, 256]):
        H = 16
        N = 1024 
        D = 64
        window_size = WINDOW_SIZE

        s = (B, H, N, D)
        q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
        k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
        v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')

        for name, fn, in methods.items():
            if "no sliding" in name.lower(): window_size = None
            lst = [fn(dt, s, window_size=window_size, qw=q, kw=k, vw=v) for _ in range(num_iters)]
            lst_time = [x[0] for x in lst]
            _time = median(lst_time)
            if i > 0 and _time > 0: method2timing[name][B] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.legend()

    # save png
    plt.savefig(f'results/a100_{window_size}_sliding_window_benchmark_batch.png', format='png', bbox_inches='tight')

    # save timings
    with open(f'results/a100_{window_size}_sliding_window_benchmark_batch.txt', 'w') as f:
        for name, timing in method2timing.items():
            f.write(f"{name}\n")
            for b, t in timing.items():
                f.write(f"{b} {t}\n")


def sliding_window_benchmark_seqlen(dt):
    num_iters = 4
    
    methods = {
        'Flash Attention (Sliding Window)': run_fa2_test, 
        'Flash Attention (no Sliding Window)': run_fa2_test, 
        'Pure PyTorch': run_torch_test, 
        'Based Sliding Window Kernel': run_based_test
    }
    method2timing = defaultdict(dict)
    for i, N in enumerate([1024, 1024, 2048, 4096, 8192, 16384, 32768]):
        H = 16
        B = 4 
        D = 64
        window_size = WINDOW_SIZE

        s = (B, H, N, D)
        q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
        k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
        v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')

        for name, fn, in methods.items():
            if "no sliding" in name.lower(): window_size = None
            lst = [fn(dt, s, window_size=window_size, qw=q, kw=k, vw=v) for _ in range(num_iters)]
            lst_time = [x[0] for x in lst]
            _time = median(lst_time)
            if i > 0 and _time > 0: method2timing[name][N] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name)
    ax.set_xlabel('Seq Len')
    ax.set_ylabel('Time (ms)')
    ax.legend()

    # save png
    plt.savefig(f'results/a100_{window_size}_sliding_window_benchmark_seqlen.png', format='png', bbox_inches='tight')

    # save timings
    with open(f'results/a100_{window_size}_sliding_window_benchmark_seqlen.txt', 'w') as f:
        for name, timing in method2timing.items():
            f.write(f"{name}\n")
            for b, t in timing.items():
                f.write(f"{b} {t}\n")
        

def sliding_window_correctness(dt):
    B = 1
    H = 1
    N = 4096 
    D = 64
    window_size = WINDOW_SIZE

    s = (B, H, N, D)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')

    torch_tot, torch_result = run_torch_test(dt, s, window_size=window_size, qw=q, kw=k, vw=v)
    based_tot, based_result = run_based_test(dt, s, window_size=window_size, qw=q, kw=k, vw=v)
    __eq("sliding window", torch_result.bfloat16(), based_result, debug=False)


print(f"Benchmarking...")
import os
os.makedirs('results', exist_ok=True)
sliding_window_benchmark_batch(torch.bfloat16)
sliding_window_benchmark_seqlen(torch.bfloat16)

print(f"\nChecking correctness...")
sliding_window_correctness(torch.bfloat16)


