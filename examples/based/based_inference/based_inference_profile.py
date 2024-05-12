import torch
import time
import based_inference as mod

from collections import defaultdict
from statistics import median
import matplotlib.pyplot as plt


def pytorch_step(kv_state, kv_state_t, k_state, q, k, v, denom: bool=True, eps: float=1e-6):
    """
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """

    # prepare inputs
    kv_state_ref = kv_state.detach().clone()
    k_state_ref  = k_state.detach().clone()
    
    # compute
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    k_state_ref += k
    kv_state_ref += torch.einsum("bf,bd->bdf", k, v)
    num = torch.einsum("bf,bdf->bd", q, kv_state_ref)
    den = torch.einsum("bf,bf->b", q, k_state_ref) + eps
    y = num / den.unsqueeze(-1)

    torch.cuda.synchronize()
    t1 = time.time()
    peak_mem = torch.cuda.max_memory_allocated(device='cuda')
    tot = t1 - t0
    return y, tot, peak_mem


def based_step_test(kv_state, kv_state_t, k_state, q, k, v):
    out_tc       = torch.zeros_like(v)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    mod.based_step(q, k, v, kv_state_t, k_state, out_tc)

    torch.cuda.synchronize()
    t1 = time.time()
    peak_mem = torch.cuda.max_memory_allocated(device='cuda')
    tot = t1 - t0
    return out_tc, tot, peak_mem


def based_step_benchmark(dt, device='cuda'):
    num_iters = 10
    
    methods = {
        'Pure PyTorch': pytorch_step, 
        'Based Kernel': based_step_test,
    }
    method2timing = defaultdict(dict)
    method2mem = defaultdict(dict)
    for b in [8, 16, 32, 64, 128, 256]:
        h = 16  
        dv = 64
        d_state = 320   # rounding 1 + 16 + 16^2 to the nearest multiple of 64 for hardware.
        print(f"{b=}, {dv=}, {h=}, {d_state=}")

        # prepare inputs 
        torch.random.manual_seed(0)
        kv_state = torch.randn(b*h, dv, d_state, dtype=dt, device=device)/d_state
        k_state = torch.randn(b*h, d_state, dtype=dt, device=device)/d_state
        v = torch.randn(b*h, dv, device=device, dtype=dt)/dv
        k = torch.randn(b*h, d_state, device=device, dtype=dt)/d_state
        q = torch.randn(b*h, d_state, device=device, dtype=dt)/d_state
        kv_state_t = kv_state.transpose(1,2).contiguous() # We prefer the other order for iteration.

        for name, fn, in methods.items():
            lst = [
                fn(kv_state=kv_state, kv_state_t=kv_state_t, k_state=k_state, v=v, k=k, q=q)
                for _ in range(num_iters)
            ]
            lst_time = [x[1] for x in lst]
            lst_mem = [x[-1] for x in lst]
            _time = median(lst_time)
            _mem = median(lst_mem)
            if b > 8: method2timing[name][b] = _time * 1000
            if b > 8: method2mem[name][b] = _mem

    print(f"Time")
    print(method2timing[name])
    print("Mem")
    print(method2mem[name])

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('based_step_benchmark.pdf', format='pdf', bbox_inches='tight')

print("Benchmarking based_step...")
based_step_benchmark(torch.bfloat16)
