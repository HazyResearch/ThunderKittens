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

def tk_forward_test(Q, K, V, causal):
    O = torch.zeros_like(Q).contiguous()
    L = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
    tk.mha_forward(Q, K, V, O, L, causal)
    return O, L

def tk_backward_test(Q, K, V, O, L, d_vec, dO, causal):
    return tk.mha_backward(Q, K, V, O, L, d_vec, dO, causal)

def check_consistency(b, h, n, d, causal, mean, std, num_iterations=100000):
    # Generate fixed inputs
    torch.manual_seed(0)
    Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')

    # Initial run to get reference outputs
    ref_O, ref_L = tk_forward_test(Q, K, V, causal)
    d_vec = torch.zeros(b, h, n, 1, device='cuda', dtype=torch.float)
    ref_qg, ref_kg, ref_vg = tk_backward_test(Q, K, V, ref_O, ref_L, d_vec, dO, causal)

    max_diff_O = 0
    max_diff_L = 0
    max_diff_qg, max_diff_kg, max_diff_vg = 0, 0, 0

    for _ in tqdm(range(num_iterations), desc="Checking consistency"):
        # Forward pass
        O, L = tk_forward_test(Q, K, V, causal)
        
        # Backward pass
        qg, kg, vg = tk_backward_test(Q, K, V, O, L, d_vec, dO, causal)
        
        # Check differences
        max_diff_O = max(max_diff_O, torch.abs(O - ref_O).max().item())
        max_diff_L = max(max_diff_L, torch.abs(L - ref_L).max().item())
        max_diff_qg = max(max_diff_qg, torch.abs(qg - ref_qg).max().item())
        max_diff_kg = max(max_diff_kg, torch.abs(kg - ref_kg).max().item())
        max_diff_vg = max(max_diff_vg, torch.abs(vg - ref_vg).max().item())
        
        if max_diff_O > 1e-7 or max_diff_L > 1e-7 or max_diff_qg > 1e-7 or max_diff_kg > 1e-7 or max_diff_vg > 1e-7:
            print(f"Iteration {_}: Large difference detected")
            print(f"Max diff O: {max_diff_O}")
            print(f"Max diff L: {max_diff_L}")
            print(f"Max diff qg: {max_diff_qg}")
            print(f"Max diff kg: {max_diff_kg}")
            print(f"Max diff vg: {max_diff_vg}")
            breakpoint()

    return max_diff_O, max_diff_L, max_diff_qg, max_diff_kg, max_diff_vg

# Example usage
b, h, n, d = 1, 16, 768, 128
causal = True
mean = 1e-1
std = 1

max_diff_O, max_diff_L, max_diff_qg, max_diff_kg, max_diff_vg = check_consistency(b, h, n, d, causal, mean, std)

print(f"Maximum difference in O across iterations: {max_diff_O}")
print(f"Maximum difference in L across iterations: {max_diff_L}")
print(f"Maximum difference in qg across iterations: {max_diff_qg}")
print(f"Maximum difference in kg across iterations: {max_diff_kg}")
print(f"Maximum difference in vg across iterations: {max_diff_vg}")