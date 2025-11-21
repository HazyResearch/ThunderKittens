import torch
from _C import 
import random
from tqdm import tqdm
from flash_attn_interface import flash_attn_func
from einops import rearrange

def pytorch_test(Q, K, V, dO, causal):
    
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    torch.backends.cuda.enable_flash_sdpa = False
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad
    
    # Q_32 = Q.to(torch.float).requires_grad_()
    # K_32 = K.to(torch.float).requires_grad_()
    # V_32 = V.to(torch.float).requires_grad_()
    # dO_32 = dO.to(torch.float)
    
    # torch.backends.cuda.enable_flash_sdpa = True
    # output = torch.nn.functional.scaled_dot_product_attention(Q_32, K_32, V_32, is_causal=causal)
    # output.backward(dO_32)
    
    # return output.detach().to(torch.bfloat16), Q_32.grad.to(torch.bfloat16), K_32.grad.to(torch.bfloat16), V_32.grad.to(torch.bfloat16)

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

def flash_attn_test(Q, K, V, dO, causal):
    
    Q  = Q.permute(0, 2, 1, 3).contiguous().detach().clone()
    K  = K.permute(0, 2, 1, 3).contiguous().detach().clone()
    V  = V.permute(0, 2, 1, 3).contiguous().detach().clone()
    dO = dO.permute(0, 2, 1, 3).contiguous().detach().clone()
    
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    # Forward pass
    output, _ = flash_attn_func(Q, K, V, causal=causal)
    
    # Backward pass
    output.backward(dO)
    
    Q_grad = Q.grad
    K_grad = K.grad
    V_grad = V.grad
    
    output = output.permute(0, 2, 1, 3).contiguous()
    
    Q_grad = Q_grad.permute(0, 2, 1, 3).contiguous()
    K_grad = K_grad.permute(0, 2, 1, 3).contiguous()
    V_grad = V_grad.permute(0, 2, 1, 3).contiguous()
    
    return output, Q_grad, K_grad, V_grad

def generate_tensor(shape, mean, std, dtype, device):
    return (torch.randn(*shape, dtype=dtype, device=device) * std + mean).contiguous()

def check_correctness(b, h, n, d, causal, mean, std, num_iterations=1000):
    print(f"Testing with b={b}, h={h}, n={n}, d={d}, causal={causal}")
    
    results = {
        'Output': {'tk_diff': 0, 'fa_diff': 0, 'sum_abs': 0, 'tk_max_diff': 0, 'fa_max_diff': 0},
        'Q grad': {'tk_diff': 0, 'fa_diff': 0, 'sum_abs': 0, 'tk_max_diff': 0, 'fa_max_diff': 0},
        'K grad': {'tk_diff': 0, 'fa_diff': 0, 'sum_abs': 0, 'tk_max_diff': 0, 'fa_max_diff': 0},
        'V grad': {'tk_diff': 0, 'fa_diff': 0, 'sum_abs': 0, 'tk_max_diff': 0, 'fa_max_diff': 0}
    }

    input_stats = {
        'Q': {'mean': 0, 'std': 0},
        'K': {'mean': 0, 'std': 0},
        'V': {'mean': 0, 'std': 0},
        'dO': {'mean': 0, 'std': 0}
    }

    for _ in tqdm(range(num_iterations), desc="Iterations"):
        seed = random.randint(0, 1000000)
        torch.manual_seed(seed)
        
        Q  = generate_tensor((b, h, n, d), mean=mean, std=std, dtype=torch.bfloat16, device='cuda')
        K  = generate_tensor((b, h, n, d), mean=mean, std=std, dtype=torch.bfloat16, device='cuda') 
        V  = generate_tensor((b, h, n, d), mean=mean, std=std, dtype=torch.bfloat16, device='cuda')
        dO = generate_tensor((b, h, n, d), mean=mean, std=std, dtype=torch.bfloat16, device='cuda')

        # Update input statistics
        for tensor_name, tensor in zip(['Q', 'K', 'V', 'dO'], [Q, K, V, dO]):
            input_stats[tensor_name]['mean'] += tensor.mean().item()
            input_stats[tensor_name]['std'] += tensor.std().item()

        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, causal)
        tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, causal)
        fa_o, fa_qg, fa_kg, fa_vg = flash_attn_test(Q, K, V, dO, causal)

        tensors = [
            ('Output', pt_o, tk_o, fa_o),
            ('Q grad', pt_qg, tk_qg, fa_qg),
            ('K grad', pt_kg, tk_kg, fa_kg),
            ('V grad', pt_vg, tk_vg, fa_vg)
        ]

        for name, pt, tk, fa in tensors:
            diff_tk = pt - tk
            diff_fa = pt - fa
            abs_diff_tk = torch.abs(diff_tk)
            abs_diff_fa = torch.abs(diff_fa)
            results[name]['tk_diff'] += torch.sum(abs_diff_tk).item()
            results[name]['fa_diff'] += torch.sum(abs_diff_fa).item()
            results[name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
            results[name]['tk_max_diff'] = max(results[name]['tk_max_diff'], torch.max(abs_diff_tk).item())
            results[name]['fa_max_diff'] = max(results[name]['fa_max_diff'], torch.max(abs_diff_fa).item())

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    # Calculate average input statistics
    for tensor_name in input_stats:
        input_stats[tensor_name]['mean'] /= num_iterations
        input_stats[tensor_name]['std'] /= num_iterations

    print("\nResults:")
    print("=" * 80)
    print(f"Input tensor statistics (averaged over {num_iterations} iterations):")
    for tensor_name, stats in input_stats.items():
        print(f"  {tensor_name}: mean = {stats['mean']:.4e}, std = {stats['std']:.4e}")
    print("-" * 80)

    for key, data in results.items():
        tol = (data['sum_abs'] / (b * h * n * d * num_iterations)) * 0.01
        tk_avg_diff = data['tk_diff'] / (b * h * n * d * num_iterations)
        fa_avg_diff = data['fa_diff'] / (b * h * n * d * num_iterations)
        
        print(f"{key}:")
        print(f"  Max acceptable avg magnitude of diff: {tol:.2e}")
        print(f"  (A) FA 3 vs PyTorch==FA2:")
        print(f"      Avg magnitude of diff: {fa_avg_diff:.2e}")
        print(f"      Max magnitude of diff: {data['fa_max_diff']:.2e}")
        print(f"      Pass: {fa_avg_diff < tol}")
        print(f"  (B) Thunderkittens vs PyTorch==FA2:")
        print(f"      Avg magnitude of diff: {tk_avg_diff:.2e}")
        print(f"      Max magnitude of diff: {data['tk_max_diff']:.2e}")
        print(f"      Pass: {tk_avg_diff < tol}")
        print("-" * 80)

        assert fa_avg_diff < tol, f"Flash Attention 3: Average difference ({fa_avg_diff}) exceeds tolerance ({tol}) for {key}"
        assert tk_avg_diff < tol, f"Thunderkittens: Average difference ({tk_avg_diff}) exceeds tolerance ({tol}) for {key}"

print("Deep Correctness Tests:")
configurations = [
    (4, 16, 768,  128, False),
    (4, 16, 768,  128, True),
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
    check_correctness(b, h, n, d, causal, mean=1e-2, std=1e-1)

print("All tests passed successfully!")