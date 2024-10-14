import torch
import thunderkittens as tk
import random
from tqdm import tqdm

def pytorch_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def h100_fwd_kernel_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    o = torch.zeros_like(Q).contiguous()
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    
    tk.mha_forward(Q, K, V, o, l_vec, causal)

    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    qg = torch.zeros_like(Q, dtype=torch.float).contiguous()
    kg = torch.zeros_like(K, dtype=torch.float).contiguous()
    vg = torch.zeros_like(V, dtype=torch.float).contiguous()

    tk.mha_backward(Q, K, V, o, l_vec, d_vec, dO, qg, kg, vg, causal)

    return o, qg, kg, vg

def check_correctness(b, h, n, d, causal, num_iterations=1000):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}, causal={causal}")
    
    results = {
        'Output': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
        'Q grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
        'K grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
        'V grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}
    }

    for _ in tqdm(range(num_iterations), desc="Iterations"):
        seed = random.randint(0, 1000000)
        torch.manual_seed(seed)
        
        Q  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        K  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        V  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        dO = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()

        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, causal)
        tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, causal)

        tensors = [
            ('Output', pt_o, tk_o),
            ('Q grad', pt_qg, tk_qg),
            ('K grad', pt_kg, tk_kg),
            ('V grad', pt_vg, tk_vg)
        ]

        for name, pt, tk in tensors:
            diff = pt - tk
            abs_diff = torch.abs(diff)
            results[name]['sum_diff'] += torch.sum(abs_diff).item()
            results[name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results[name]['max_diff'] = max(results[name]['max_diff'], torch.max(abs_diff).item())

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    for key, data in results.items():
        tol = (data['sum_abs'] / (b * h * n * d * num_iterations)) * 0.01
        avg_diff = data['sum_diff'] / (b * h * n * d * num_iterations)
        max_diff = data['max_diff']
        
        print(f"{key}:")
        print(f"  Max acceptable avg magnitude of diff: {tol:.10f}")
        print(f"  Avg magnitude of diff: {avg_diff:.10f}")
        print(f"  Max magnitude of diff: {max_diff:.10f}")
        print(f"  Pass: {avg_diff < tol}")
        print("-" * 40)

        assert avg_diff < tol, f"Average difference ({avg_diff}) exceeds tolerance ({tol}) for {key}"

print("Deep Correctness Tests:")
configurations = [
    (4, 16, 768,    128, False),
    (4, 16, 768*2,  128, False),
    (4, 16, 768*4,  128, False),
    (4, 16, 768*8,  128, False),
    (4, 16, 768*16, 128, False),
    (4, 16, 768,    128, True),
    (4, 16, 768*2,  128, True),
    (4, 16, 768*4,  128, True),
    (4, 16, 768*8,  128, True),
    (4, 16, 768*16, 128, True),
    (4, 32, 768,    64,  False),
    (4, 32, 768*2,  64,  False),
    (4, 32, 768*4,  64,  False),
    (4, 32, 768*8,  64,  False),
    (4, 32, 768*16, 64,  False),
    (4, 32, 768,    64,  True),
    (4, 32, 768*2,  64,  True),
    (4, 32, 768*4,  64,  True),
    (4, 32, 768*8,  64,  True),
    (4, 32, 768*16, 64,  True),
]

for b, h, n, d, causal in configurations:
    check_correctness(b, h, n, d, causal)

print("All tests passed successfully!")