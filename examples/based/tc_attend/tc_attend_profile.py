import torch
debug=False
from test_build_utils import __eq
import tc_attend as mod
dtypes = [torch.bfloat16]

import torch
from flash_attn import flash_attn_with_kvcache
import time
from statistics import median
from collections import defaultdict
import matplotlib.pyplot as plt


def run_fa2_test(dt,s, window_size=64, j=100):

    # prep data
    b,h,n,d = s
    k_state = torch.zeros((b, n, h, d), dtype=dt, device='cuda')
    v_state = torch.randn((b, n, h, d), dtype=dt, device='cuda')
    q = torch.randn(b, 1, h, d, dtype=dt, device='cuda')

    if window_size is not None:

        # benchmark
        torch.cuda.synchronize()
        t0 = time.time()
        result = flash_attn_with_kvcache(
            q, 
            k_state, 
            v_state,
            # q, SA: exclude the state update here
            # q,
            cache_seqlens=j,
            softmax_scale=0.5,
            causal=True, 
            window_size=(window_size,window_size),
        )
        torch.cuda.synchronize()
        t1 = time.time()
    else:

        # benchmark
        torch.cuda.synchronize()
        t0 = time.time()
        result = flash_attn_with_kvcache(
            q, 
            k_state, 
            v_state,
            # q, SA: exclude the state update here
            # q,
            cache_seqlens=j,
            softmax_scale=0.5,
            causal=True, 
        )
        torch.cuda.synchronize()
        t1 = time.time()
    tot = t1-t0
    return tot, result


def run_torch_test(dt, s, window_size=64, j=100, qw=None, kw=None, vw=None):
    
    # prep data
    b,h,n,d = s
    if qw is None:
        q,k,v = [torch.randn(s, device='cuda', dtype=dt)/d*h for i in range(3)]
        qw = q[:,:,j].unsqueeze(2).contiguous()
        kw = k[:,:,j-window_size:j].contiguous()
        vw = v[:,:,j-window_size:j].contiguous()
        
    # benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    w = torch.einsum("bhod,bhnd->bhn",qw, kw)
    a = torch.nn.functional.softmax(w, dim=-1)
    result = torch.einsum("bhn,bhnd->bhd", a, vw)
    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0
    return tot, result


def run_based_test(dt, s, window_size=64, j=100, qw=None, kw=None, vw=None):

    # prep data
    b,h,n,d = s
    if qw is None:
        q,k,v = [torch.randn(s, device='cuda', dtype=dt)/d*h for i in range(3)]
        kw = k[:,:,j-window_size:j].contiguous()
        vw = v[:,:,j-window_size:j].contiguous()
        qw = q[:,:,j].unsqueeze(2).contiguous()
    o     = torch.zeros((b,h,d), dtype=dt, device='cuda')

    # benchmark kernel
    torch.cuda.synchronize()
    t0 = time.time()
    mod.sliding_window(j, qw, kw, vw, o)
    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1-t0
    return tot, o


def sliding_window_benchmark(dt):
    num_iters = 4
    j_vals = [100, 250, 500, 750]
    
    methods = {
        'Flash Attention (Sliding Window)': run_fa2_test, 
        'Flash Attention (no Sliding Window)': run_fa2_test, 
        'Pure PyTorch': run_torch_test, 
        'Based Sliding Window Kernel': run_based_test
    }
    method2timing = defaultdict(dict)
    for b in [16, 32, 64, 128, 256]:
        h = 16
        n = 1024 
        d = 64
        s = (b,h,n,d)
        print(f"{b=}, {n=}, {d=}, {h=}")

        for name, fn, in methods.items():
            window_size = 64
            if name == 'fa': window_size = None
            lst = [fn(dt, s, window_size=window_size, j=j) for _, j in zip(range(num_iters), j_vals)]
            lst_time = [x[0] for x in lst]
            _time = median(lst_time)
            method2timing[name][b] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('h100_sliding_window_benchmark.pdf', format='pdf', bbox_inches='tight')
        

def sliding_window_correctness(dt):
    b = 1
    h = 1
    n = 128 
    d = 64

    window_size = 64
    j = 85

    s = (b,h,n,d)
    q,k,v = [torch.ones(s, device='cuda', dtype=dt)/d*h for i in range(3)]
    fill_row = 60
    q[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)
    k[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)
    v[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)
    fill_row= 61
    q[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)
    k[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)+100
    v[:,:,fill_row] = torch.arange(0, d, device='cuda', dtype=dt)+100

    qw = q[:,:,j].unsqueeze(2).contiguous()
    kw = k[:,:,j-window_size:j].contiguous()
    vw = v[:,:,j-window_size:j].contiguous()

    torch_tot, torch_result = run_torch_test(dt, s, window_size=64, qw=qw, kw=kw, vw=vw)
    based_tot, based_result = run_based_test(dt, s, window_size=64, qw=qw, kw=kw, vw=vw)
    __eq("sliding window", torch_result.bfloat16(), based_result, debug=False)

print(f"Benchmarking...")
sliding_window_benchmark(torch.bfloat16)
print(f"\nChecking correctness...")
sliding_window_correctness(torch.bfloat16)