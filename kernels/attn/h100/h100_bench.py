import torch
import numpy as np
import h100 as tk_train

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def benchmark_attention(configurations):
    for B, H, N, D in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}")

        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')

        o = torch.zeros_like(q)
        grad_output = torch.randn_like(q, requires_grad=False)

        qg = torch.zeros_like(q, requires_grad=False, dtype=torch.float32)
        kg = torch.zeros_like(k, requires_grad=False, dtype=torch.float32)
        vg = torch.zeros_like(v, requires_grad=False, dtype=torch.float32)

        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)
        d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)

        # Warmup for forward pass
        for _ in range(10):
            tk_train.attention_forward(q, k, v, o, l_vec, False)

        # Prepare for timing forward pass
        start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

        # Time the forward pass
        for i in range(100):
            o.zero_()
            
            torch.cuda.synchronize()
            start_events_fwd[i].record()
            
            tk_train.attention_forward(q, k, v, o, l_vec, False)
            
            torch.cuda.synchronize()
            end_events_fwd[i].record()

        torch.cuda.synchronize()
        times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        time_us_fwd = np.mean(times_fwd) * 1000

        print(f"Average time for forward pass in us: {time_us_fwd:.2f}")
        print(f"Average efficiency for forward pass in TFLOPS: {efficiency(flops(B, N, H, D, False, 'fwd'), time_us_fwd)}")
        print("-" * 60)
        # Warmup for backward pass
        for _ in range(10):
            o.zero_()
            l_vec.zero_()
            
            tk_train.attention_forward(q, k, v, o, l_vec, False)
            
            # zero out gradients
            qg.zero_()
            kg.zero_()
            vg.zero_()
            d_vec.zero_()
            
            tk_train.attention_backward(q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg, False)

        # Prepare for timing backward pass
        start_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        end_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

        # Time the backward pass
        for i in range(100):
            o.zero_()
            l_vec.zero_()
            
            tk_train.attention_forward(q, k, v, o, l_vec, False)
            
            # zero out gradients
            qg.zero_()
            kg.zero_()
            vg.zero_()
            d_vec.zero_()
            
            torch.cuda.synchronize()
            start_events_bwd[i].record()
            
            tk_train.attention_backward(q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg, False)
            
            torch.cuda.synchronize()
            end_events_bwd[i].record()

        torch.cuda.synchronize()
        times_bwd = [s.elapsed_time(e) for s, e in zip(start_events_bwd, end_events_bwd)]
        time_us_bwd = np.mean(times_bwd) * 1000

        print(f"Average time for backward pass in us: {time_us_bwd:.2f}")
        print(f"Average efficiency for backward pass in TFLOPS: {efficiency(flops(B, N, H, D, False, 'bwd'), time_us_bwd)}")
        print("=" * 60)
        
# Example list of configurations to test
configurations = [
    (32, 16, 768, 128),
    (16, 16, 768*2, 128),
    (8, 16, 768*4, 128),
    (4, 16, 768*8, 128), 
    (2, 16, 768*16, 128), 
    (1, 16, 768*32, 128)
]

benchmark_attention(configurations)
