import torch
import thunderkittens as tk
import numpy as np
from tqdm import tqdm

from flash_attn_interface import flash_attn_func

def generate_tensor(shape, mean, std, dtype, device):
    
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()

def tk_backward_test(Q, K, V, O, L, d_vec, dO, causal):
    
    return tk.mha_backward(Q, K, V, O, L, d_vec, dO, causal)

def check_consistency(b, h, n, d, causal, mean, std, num_iterations=100000):
    
    # Generate fixed inputs
    torch.manual_seed(0)
    Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    O  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    
    _, L  = flash_attn_func(Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3), causal=causal)
    d_vec = torch.zeros(b, h, n, 1, device='cuda', dtype=torch.float)

    # Initial run to get reference outputs
    ref_qg, ref_kg, ref_vg = tk_backward_test(Q, K, V, O, L, d_vec, dO, causal)

    max_diff_qg, max_diff_kg, max_diff_vg = 0, 0, 0

    for _ in tqdm(range(num_iterations), desc="Checking consistency"):
        qg, kg, vg = tk_backward_test(Q, K, V, O, L, d_vec, dO, causal)
        
        max_diff_qg = torch.abs(qg - ref_qg).max()
        max_diff_kg = torch.abs(kg - ref_kg).max()
        max_diff_vg = torch.abs(vg - ref_vg).max()
        
        if max_diff_qg > 1e-7 or max_diff_kg > 1e-7 or max_diff_vg > 1e-7:
            breakpoint()

    return max_diff_qg, max_diff_kg, max_diff_vg

# Example usage
b, h, n, d = 1, 16, 768, 128
causal = True
mean = 1e-1
std = 1

max_diff_qg, max_diff_kg, max_diff_vg = check_consistency(b, h, n, d, causal, mean, std)