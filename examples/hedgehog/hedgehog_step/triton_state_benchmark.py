import torch
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from triton_state_update import hedgehog_step, hedgehog_step_ref


def step_benchmark(dt, device='cuda'):
    num_iters = 10
    
    methods = {
        'Pure PyTorch': hedgehog_step_ref, 
        'Triton Kernel': hedgehog_step,
    }
    method2timing = defaultdict(dict)
    method2mem = defaultdict(dict)
    for i, b in enumerate([2, 8, 16, 32, 64, 128, 256]):
        h = 16  
        dv = 64
        d_state = 256   
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
            torch.cuda.synchronize()
            t0 = time.time()
            
            outs = [
                fn(kv_state=kv_state, k_state=k_state, v=v, k=k, q=q)
                for _ in range(num_iters)
            ]

            torch.cuda.synchronize()
            t1 = time.time()
            _time = (t1 - t0) / num_iters

            if i != 0: # triton is slow on the first call
                method2timing[name][b] = _time * 1000

    print(f"Time")
    print(method2timing[name])

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('step_benchmark.png')


print("Benchmarking step...")
step_benchmark(torch.bfloat16)