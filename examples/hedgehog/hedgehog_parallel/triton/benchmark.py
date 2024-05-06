import torch
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from hedgehog import HedgehogBased


def parallel_benchmark(dt=torch.bfloat16, device='cuda'):
    num_iters = 20
    
    method2timing = defaultdict(dict)
    method2mem = defaultdict(dict)
    for i, seq_len in enumerate([256, 512, 1024, 2048, 4096]):
        batch_size = 6
        n_heads = 4

        d  = 64     # this is the head dimension and input dimension
        dv = 128    # this is the feature dimension
        
        print(f"B={batch_size}, H={n_heads}, N={seq_len}, D={d}, DV={dv}")
        print("-----------------------------------")

        # prepare inputs 
        torch.random.manual_seed(0)
       
        q = (torch.randn(batch_size, n_heads, seq_len, d).cuda().to(torch.bfloat16) / d).requires_grad_(True)
        k = (torch.randn(batch_size, n_heads, seq_len, d).cuda().to(torch.bfloat16) / d).requires_grad_(True)
        v = (torch.randn(batch_size, n_heads, seq_len, dv).cuda().to(torch.bfloat16) / dv).requires_grad_(True)
        
        # initialize model
        scaling, norm = False, False
        
        # manual model
        torch.manual_seed(1)
        model_ref = HedgehogBased(num_heads=n_heads, head_dim=d, feature_dim=dv, input_dim=d, dtype=torch.bfloat16, zero_init=False, use_triton=False)
        ref = model_ref(q, k, v, scaling, norm)

        # triton model
        torch.manual_seed(1)
        model_tri = HedgehogBased(num_heads=n_heads, head_dim=d, feature_dim=dv, input_dim=d, dtype=torch.bfloat16, zero_init=False, use_triton=True)
        tri = model_tri(q, k, v, scaling, norm)

        # fast transformers model
        torch.manual_seed(1)
        model_ft = HedgehogBased(num_heads=n_heads, head_dim=d, feature_dim=dv, input_dim=d, dtype=torch.bfloat16, zero_init=False, use_triton=False, use_fast_transformers=True)
        ft = model_ft(q, k, v, scaling, norm)

        # ref model ---
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            
            outs = [
                model_ref(q, k, v, scaling, norm)
                for _ in range(num_iters)
            ]

            torch.cuda.synchronize()
            t1 = time.time()
            _time = (t1 - t0) / num_iters

            if i != 0: # triton is slow on the first call
                method2timing['Pure PyTorch'][seq_len] = _time * 1000
        except:
            print("Pure PyTorch failed")
            pass
            
        # triton model ---
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            
            outs = [
                model_tri(q, k, v, scaling, norm)
                for _ in range(num_iters)
            ]

            torch.cuda.synchronize()
            t1 = time.time()
            _time = (t1 - t0) / num_iters

            if i != 0: # triton is slow on the first call
                method2timing['Triton Kernel'][seq_len] = _time * 1000
        except:
            print("Triton Kernel failed")
            pass

        # fast transformers model ---
        try:
            torch.cuda.synchronize()
            t0 = time.time()

            outs = [
                model_ft(q, k, v, scaling, norm)
                for _ in range(num_iters)
            ]

            torch.cuda.synchronize()
            t1 = time.time()
            _time = (t1 - t0) / num_iters

            if i != 0: # triton is slow on the first call
                method2timing['Fast Transformers'][seq_len] = _time * 1000
        except:
            print("Fast Transformers failed")
            pass


    # plot: time vs. sequence length
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('parallel_benchmark.png')


print("Benchmarking parallel...")
parallel_benchmark()

