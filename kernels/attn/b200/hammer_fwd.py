import torch
import thunderkittens as tk
from tqdm import tqdm

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

def check_consistency(b, h, n, d, causal, mean, std, num_iterations=100000):
    # Generate fixed inputs
    torch.manual_seed(0)
    Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')

    # Initial run to get reference outputs
    ref_O, ref_L = tk_forward_test(Q, K, V, causal)

    max_diff_O = 0
    max_diff_L = 0

    for i in tqdm(range(num_iterations), desc="Checking consistency"):
        # Forward pass
        O, L = tk_forward_test(Q, K, V, causal)
        
        # Check differences
        current_diff_O = torch.abs(O - ref_O).max().item()
        current_diff_L = torch.abs(L - ref_L).max().item()
        
        max_diff_O = max(max_diff_O, current_diff_O)
        max_diff_L = max(max_diff_L, current_diff_L)
        
        if current_diff_O > 1e-7 or current_diff_L > 1e-7:
            print(f"Iteration {i}: Large difference detected")
            print(f"Current diff O: {current_diff_O}")
            print(f"Current diff L: {current_diff_L}")
            # breakpoint()

    return max_diff_O, max_diff_L

# Example usage
b, h, n, d = 1, 16, 768*8, 128
causal = True
mean = 1e-1
std = 1

max_diff_O, max_diff_L = check_consistency(b, h, n, d, causal, mean, std)

print(f"Maximum difference in O across iterations: {max_diff_O}")
print(f"Maximum difference in L across iterations: {max_diff_L}")