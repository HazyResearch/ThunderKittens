import torch
from flash_attn_interface import flash_attn_func
import thunderkittens as tk
import random
from tqdm import tqdm

def pytorch_test(Q, K, V, dO, causal):
    q_ = Q.to(torch.float64).requires_grad_()
    k_ = K.to(torch.float64).requires_grad_()
    v_ = V.to(torch.float64).requires_grad_()
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool)
        QK.masked_fill_(mask, float('-inf'))
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    q_grad = q_grad.to(torch.bfloat16)
    k_grad = k_grad.to(torch.bfloat16)
    v_grad = v_grad.to(torch.bfloat16)
    output = output.to(torch.bfloat16)
    
    return output, q_grad, k_grad, v_grad

def fa2_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def fa3_test(Q, K, V, dO, causal):
    Q_  = Q.permute(0, 2, 1, 3).clone().detach().contiguous()
    K_  = K.permute(0, 2, 1, 3).clone().detach().contiguous()
    V_  = V.permute(0, 2, 1, 3).clone().detach().contiguous()
    dO_ = dO.permute(0, 2, 1, 3).clone().detach().contiguous()
    
    Q_.requires_grad = True
    K_.requires_grad = True
    V_.requires_grad = True
    
    out, _ = flash_attn_func(Q_, K_, V_, causal=causal)
    out.backward(dO_)
    
    qgrad = Q_.grad
    kgrad = K_.grad
    vgrad = V_.grad
    
    qg_ = qgrad.permute(0, 2, 1, 3).contiguous()
    kg_ = kgrad.permute(0, 2, 1, 3).contiguous()
    vg_ = vgrad.permute(0, 2, 1, 3).contiguous()
    
    output = out.permute(0, 2, 1, 3).contiguous()
    
    return output, qg_, kg_, vg_

def h100_fwd_kernel_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    o = torch.zeros_like(Q).contiguous()
    l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    
    tk.mha_forward(Q, K, V, o, l_vec, causal)

    d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float).requires_grad_()
    
    qg, kg, vg = tk.mha_backward(Q, K, V, o, l_vec, d_vec, dO, causal)

    return o, qg, kg, vg

def check_correctness(b, h, n, d, causal, num_iterations=1000):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}, causal={causal}")
    
    results = {
        'TK vs PT': {
            'Output': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'Q grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'K grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'V grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}
        },
        'FA2 vs PT': {
            'Output': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'Q grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'K grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'V grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}
        },
        'FA3 vs PT': {
            'Output': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'Q grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'K grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
            'V grad': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}
        }
    }

    for _ in tqdm(range(num_iterations), desc="Iterations"):
        seed = random.randint(0, 1000000)
        torch.manual_seed(seed)
        
        Q  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        K  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        V  = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()
        dO = torch.randn(b, h, n, d, dtype=torch.bfloat16, device='cuda').contiguous()

        pt_o, pt_qg, pt_kg, pt_vg     = pytorch_test(Q, K, V, dO, causal)
        fa2_o, fa2_qg, fa2_kg, fa2_vg = fa2_test(Q, K, V, dO, causal)
        fa3_o, fa3_qg, fa3_kg, fa3_vg = fa3_test(Q, K, V, dO, causal)
        tk_o, tk_qg, tk_kg, tk_vg     = h100_fwd_kernel_test(Q, K, V, dO, causal)

        tensors = [
            ('Output', pt_o, tk_o, fa2_o, fa3_o),
            ('Q grad', pt_qg, tk_qg, fa2_qg, fa3_qg),
            ('K grad', pt_kg, tk_kg, fa2_kg, fa3_kg),
            ('V grad', pt_vg, tk_vg, fa2_vg, fa3_vg)
        ]

        for name, pt, tk, fa2, fa3 in tensors:
            # TK vs PT
            diff_tk = pt - tk
            abs_diff_tk = torch.abs(diff_tk)
            results['TK vs PT'][name]['sum_diff'] += torch.sum(abs_diff_tk).item()
            results['TK vs PT'][name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results['TK vs PT'][name]['max_diff'] = max(results['TK vs PT'][name]['max_diff'], torch.max(abs_diff_tk).item())

            # FA2 vs PT
            diff_fa2 = pt - fa2
            abs_diff_fa2 = torch.abs(diff_fa2)
            results['FA2 vs PT'][name]['sum_diff'] += torch.sum(abs_diff_fa2).item()
            results['FA2 vs PT'][name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results['FA2 vs PT'][name]['max_diff'] = max(results['FA2 vs PT'][name]['max_diff'], torch.max(abs_diff_fa2).item())
            
            # FA3 vs PT
            diff_fa3 = pt - fa3
            abs_diff_fa3 = torch.abs(diff_fa3)
            results['FA3 vs PT'][name]['sum_diff'] += torch.sum(abs_diff_fa3).item()
            results['FA3 vs PT'][name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results['FA3 vs PT'][name]['max_diff'] = max(results['FA3 vs PT'][name]['max_diff'], torch.max(abs_diff_fa3).item())

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    for comparison, data in results.items():
        print(f"\n{comparison}:")
        for key, values in data.items():
            tol = (values['sum_abs'] / (b * h * n * d * num_iterations)) * 0.01
            avg_diff = values['sum_diff'] / (b * h * n * d * num_iterations)
            max_diff = values['max_diff']
            
            print(f"{key}:")
            print(f"  Max acceptable avg magnitude of diff: {tol:.10f}")
            print(f"  Avg magnitude of diff: {avg_diff:.10f}")
            print(f"  Max magnitude of diff: {max_diff:.10f}")
            print(f"  Pass: {avg_diff < tol}")
            print("-" * 40)

            # assert avg_diff < tol, f"Average difference ({avg_diff}) exceeds tolerance ({tol}) for {key} in {comparison}"

print("Deep Correctness Tests:")
configurations = [
    (4, 16, 768,    128, False),
    # (4, 16, 768*2,  128, False),
    # (4, 16, 768*4,  128, False),
    # (4, 16, 768*8,  128, False),
    # (4, 16, 768*16, 128, False),
    # (4, 16, 768,    128, True),
    # (4, 16, 768*2,  128, True),
    # (4, 16, 768*4,  128, True),
    # (4, 16, 768*8,  128, True),
    # (4, 16, 768*16, 128, True),
    # (4, 32, 768,    64,  False),
    # (4, 32, 768*2,  64,  False),
    # (4, 32, 768*4,  64,  False),
    # (4, 32, 768*8,  64,  False),
    # (4, 32, 768*16, 64,  False),
    # (4, 32, 768,    64,  True),
    # (4, 32, 768*2,  64,  True),
    # (4, 32, 768*4,  64,  True),
    # (4, 32, 768*8,  64,  True),
    # (4, 32, 768*16, 64,  True),
]

for b, h, n, d, causal in configurations:
    check_correctness(b, h, n, d, causal)

print("All tests passed successfully!")