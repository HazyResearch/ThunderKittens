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

    max_diff_O = 0
    max_diff_L = 0
    max_diff_qg, max_diff_kg, max_diff_vg = 0, 0, 0
    threshold = 1e-5
    for _ in tqdm(range(num_iterations), desc="Checking consistency"):
        Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        Q.requires_grad, K.requires_grad, V.requires_grad = True, True, True

        high_precision_Q, high_precision_K, high_precision_V = Q.clone().detach().to(torch.float64), K.clone().detach().to(torch.float64), V.clone().detach().to(torch.float64)
        high_precision_Q.requires_grad, high_precision_K.requires_grad, high_precision_V.requires_grad = True, True, True
        high_precision_attn = torch.einsum('bhnd,bhmd->bhnm', high_precision_Q, high_precision_K)
        high_precision_attn/=d**.5
        if causal:
            causal_mask = torch.tril(torch.ones_like(high_precision_attn), diagonal=0)
            high_precision_attn = high_precision_attn.masked_fill(causal_mask == 0, float('-inf'))
        high_precision_attn = torch.nn.functional.softmax(high_precision_attn, dim=-1)
        high_precision_o = torch.einsum('bhnm,bhmd->bhnd', high_precision_attn, high_precision_V).to(torch.bfloat16)
        high_precision_o.backward(dO)
        high_precision_qg = high_precision_Q.grad.to(torch.bfloat16)
        high_precision_kg = high_precision_K.grad.to(torch.bfloat16)
        high_precision_vg = high_precision_V.grad.to(torch.bfloat16)
        high_precision_Q.grad, high_precision_K.grad, high_precision_V.grad = None, None, None

        ref_O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        ref_O.backward(dO)
        ref_qg = Q.grad.to(torch.bfloat16)
        ref_kg = K.grad.to(torch.bfloat16)
        ref_vg = V.grad.to(torch.bfloat16)
        Q.grad, K.grad, V.grad = None, None, None

        # Forward pass
        O, L = tk_forward_test(Q.clone().detach(), K.clone().detach(), V.clone().detach(), causal)
        
        # Backward pass
        d_vec = torch.zeros(b, h, n, 1, device='cuda', dtype=torch.float)
        qg, kg, vg = tk_backward_test(Q.clone().detach(), K.clone().detach(), V.clone().detach(), O.clone().detach(), L.clone().detach(), d_vec, dO, causal)
        
        # Check differences
        max_diff_O = torch.abs(O - high_precision_o).max().item()
        max_diff_qg = torch.abs(qg - high_precision_qg).max().item()
        max_diff_kg = torch.abs(kg - high_precision_kg).max().item()
        max_diff_vg = torch.abs(vg - high_precision_vg).max().item()
        
        if max_diff_O > threshold or max_diff_L > threshold or max_diff_qg > threshold or max_diff_kg > threshold or max_diff_vg > threshold:
            print(f"Iteration {_}: Large difference detected")
            print(f"Max diff O: {max_diff_O}, Avg diff O: {torch.abs(O - high_precision_o).mean().item()}")
            print(f"Max diff qg: {max_diff_qg}, Avg diff qg: {torch.abs(qg - high_precision_qg).mean().item()}")
            print(f"Max diff kg: {max_diff_kg}, Avg diff kg: {torch.abs(kg - high_precision_kg).mean().item()}")
            print(f"Max diff vg: {max_diff_vg}, Avg diff vg: {torch.abs(vg - high_precision_vg).mean().item()}")
            print(f"Max FA2 diff O: {torch.abs(ref_O - high_precision_o).max().item()}, Avg diff O: {torch.abs(ref_O - high_precision_o).mean().item()}")
            print(f"Max FA2 diff qg: {torch.abs(ref_qg - high_precision_qg).max().item()}, Avg diff qg: {torch.abs(ref_qg - high_precision_qg).mean().item()}")
            print(f"Max FA2 diff kg: {torch.abs(ref_kg - high_precision_kg).max().item()}, Avg diff kg: {torch.abs(ref_kg - high_precision_kg).mean().item()}")
            print(f"Max FA2 diff vg: {torch.abs(ref_vg - high_precision_vg).max().item()}, Avg diff vg: {torch.abs(ref_vg - high_precision_vg).mean().item()}")
            breakpoint()

    return max_diff_O, max_diff_L, max_diff_qg, max_diff_kg, max_diff_vg

# Example usage
b, h, n, d = 12, 12, 768, 64
causal = True
mean = 15
std = 200

max_diff_O, max_diff_L, max_diff_qg, max_diff_kg, max_diff_vg = check_consistency(b, h, n, d, causal, mean, std)

print(f"Maximum difference in O across iterations: {max_diff_O}")
print(f"Maximum difference in L across iterations: {max_diff_L}")
print(f"Maximum difference in qg across iterations: {max_diff_qg}")
print(f"Maximum difference in kg across iterations: {max_diff_kg}")
print(f"Maximum difference in vg across iterations: {max_diff_vg}")