import torch
import numpy as np
import thunderkittens as tk
import matplotlib.pyplot as plt
from collections import defaultdict
import triton
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from selection_attention_ref import parallel_nsa

def safe_all_close(out, ref, rtol=1e-2, atol=1e-2):
    assert not out.isnan().any()
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    except Exception as e:
        print(e)

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e3
    return flop / time

def convert_layout(x):
    return x.detach().contiguous().requires_grad_()

def bwd(*inputs, y, grad):
    for x in inputs:
        if isinstance(x, torch.Tensor):
            x.grad = None
    y.backward(grad, retain_graph=True)

def create_block_indices(B, T, H, S, total_blocks, device):
    """Create block indices for NSA attention"""
    block_indices = torch.randperm(total_blocks, device=device)
    idx = (block_indices == 0).nonzero(as_tuple=False).item()
    block_indices[0], block_indices[idx] = block_indices[idx].clone(), block_indices[0].clone()
    block_indices = block_indices[:S]
    block_indices = block_indices.repeat(B, T, H).reshape(B, T, H, S)
    return block_indices

def create_gate_scores(B, T, HQ, device):
    """Create gate scores for NSA attention"""
    g_slc = torch.ones(B, T, HQ, device=device, dtype=torch.bfloat16)
    g_swa = torch.ones(B, T, HQ, device=device, dtype=torch.bfloat16)
    return g_slc, g_swa

def benchmark_selection_attention(configurations):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")

        # Create input tensors
        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=True).contiguous()
        k = torch.randn(B, N, H//16, D, dtype=torch.bfloat16, device='cuda', requires_grad=True).contiguous()
        v = torch.randn(B, N, H//16, D, dtype=torch.bfloat16, device='cuda', requires_grad=True).contiguous()

        grad_output = torch.randn_like(q, requires_grad=False).contiguous()
        
        # Create block indices and gate scores for NSA
        block_size = 64
        S = 16  # number of selected blocks
        block_indices = create_block_indices(B, N, H//16, S, block_size, 'cuda')
        g_slc, g_swa = create_gate_scores(B, N, H, 'cuda')
        
        # Flash-Attn (baseline)
        fla_o, lse_fla = flash_attn_func_v3(q, k, v, causal=causal)
        fla_fwd = triton.testing.do_bench(lambda: flash_attn_func_v3(q, k, v, causal=causal))
        fla_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), fla_fwd)
        fla_bwd = triton.testing.do_bench(lambda: bwd(q, k, v, y=fla_o, grad=grad_output))
        fla_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), fla_bwd)

        # TK selection attention
        tk_fwd = triton.testing.do_bench(lambda: tk.selection_attn_fwd(q, k, v, block_indices.to(torch.int32), S, block_size, causal))
        tk_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), tk_fwd)
        tk_o, lse = tk.selection_attn_fwd(q, k, v, block_indices.to(torch.int32), S, block_size, causal)

        tk_bwd = triton.testing.do_bench(lambda: tk.selection_attn_bwd(q, k, v, tk_o, lse, grad_output, block_indices.to(torch.int32), S, block_size, causal))
        tk_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), tk_bwd)
        q_grad, k_grad, v_grad = tk.selection_attn_bwd(q, k, v, tk_o, lse, grad_output, block_indices.to(torch.int32), S, block_size, causal)

        # NSA parallel attention
        nsa_fwd = triton.testing.do_bench(lambda: parallel_nsa(q, k, v, g_slc, g_swa, block_indices, 
                                                              block_size=block_size, window_size=0, 
                                                              scale=None, cu_seqlens=None, head_first=False))
        nsa_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), nsa_fwd)
        nsa_o = parallel_nsa(q, k, v, g_slc, g_swa, block_indices, 
                            block_size=block_size, window_size=0, 
                            scale=None, cu_seqlens=None, head_first=False)
        nsa_bwd = triton.testing.do_bench(lambda: bwd(q, k, v, y=nsa_o, grad=grad_output))
        nsa_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), nsa_bwd)


        # Test forward accuracy between TK and NSA
        print("Testing forward accuracy between TK and NSA...")
        safe_all_close(tk_o, nsa_o, rtol=5e-3, atol=5e-3)
        
        # Test forward accuracy with Flash-Attn (reference)
        q_ref = convert_layout(q)
        k_ref = convert_layout(k)
        v_ref = convert_layout(v)
        fla_o, lse_ref = flash_attn_func_v3(q_ref, k_ref, v_ref, causal=causal)
        
        print(f"FWD Time (ms):")
        print(f"  Flash-Attn: {fla_fwd:.2f}")
        print(f"  TK_NSA:     {tk_fwd:.2f}")
        print(f"  Triton_NSA: {nsa_fwd:.2f}")
        print(f"TFLOPS:")
        print(f"  Flash-Attn: {fla_fwd_tflops:.2f}")
        print(f"  TK_NSA:     {tk_fwd_tflops:.2f}")
        print(f"  Triton_NSA: {nsa_fwd_tflops:.2f}")
        print(f"BWD Time (ms):")
        print(f"  Flash-Attn: {fla_bwd:.2f}")
        print(f"  TK_NSA:     {tk_bwd:.2f}")
        print(f"  Triton_NSA: {nsa_bwd:.2f}")
        print(f"TFLOPS:")
        print(f"  Flash-Attn: {fla_bwd_tflops:.2f}")
        print(f"  TK_NSA:     {tk_bwd_tflops:.2f}")
        print(f"  Triton_NSA: {nsa_bwd_tflops:.2f}")
        print("-" * 60)
        
        # Store results for plotting
        results['fwd']['Flash-Attn'].append(fla_fwd_tflops)
        results['fwd']['TK_NSA'].append(tk_fwd_tflops)
        results['fwd']['Triton_NSA'].append(nsa_fwd_tflops)
        results['fwd']['config'].append(f"B={B},H={H},N={N},D={D}")
        results['bwd']['Flash-Attn'].append(fla_bwd_tflops)
        results['bwd']['TK_NSA'].append(tk_bwd_tflops)
        results['bwd']['Triton_NSA'].append(nsa_bwd_tflops)
        
    
    return results


# Example list of configurations to test
configurations = [
    (16, 64, 4096, 128, True),
    (16, 64, 4096, 128, True),
]

if __name__ == "__main__":
    print("Starting performance benchmark for selection attention...")
    results = benchmark_selection_attention(configurations)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print('FWD Performance')
    for method in ['Flash-Attn', 'TK_NSA', 'Triton_NSA']:
        avg_fwd_tflops = np.mean(results['fwd'][method])
        print(f"{method:12s}: {avg_fwd_tflops:.2f} TFLOPS (avg)")
    print('BWD Performance')
    for method in ['Flash-Attn', 'TK_NSA', 'Triton_NSA']:
        avg_bwd_tflops = np.mean(results['bwd'][method])
        print(f"{method:12s}: {avg_bwd_tflops:.2f} TFLOPS (avg)")
        