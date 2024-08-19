import torch
import numpy as np

torch.manual_seed(42)

# Constants
B = 1     # Batch size
N = 4096    # Sequence length
D_QK = 64   # Query/Key dimension
D_VO = 128  # Value/Output dimension
COLLABORATIVE_SMS = 1  # Number of collaborative SMs
EXPANSION_FACTOR = COLLABORATIVE_SMS*2

def create_terraced_lower_triangular(size, block_size=64):
    # Create a lower triangular matrix of ones
    matrix = torch.tril(torch.ones(size, size, device='cuda:0'))
    
    # Create a mask for the block diagonal
    mask = torch.zeros(size, size, dtype=torch.bool, device='cuda:0')
    for i in range(0, size, block_size):
        end = min(i + block_size, size)
        mask[i:end, i:end] = True
    
    # Apply the mask to cut out the block diagonal
    matrix[mask] = 0
    
    return matrix

def initialize_tensors():
    # Initialize Q, K, V tensors
    q = torch.randn(B, N, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    k = torch.randn(B, N, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    v = torch.randn(B, N, D_VO, dtype=torch.bfloat16, device='cuda:0')

    # Initialize Q_map and K_map tensors
    q_map = torch.randn(EXPANSION_FACTOR, D_QK, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    k_map = torch.randn(EXPANSION_FACTOR, D_QK, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK

    return q, k, v, q_map, k_map

def compute_output(q, k, v, q_map, k_map, mask):
    q_heads = torch.relu(torch.einsum('edk,bnd->benk', q_map, q)).to(torch.bfloat16)
    k_heads = torch.relu(torch.einsum('edk,bnd->benk', k_map, k)).to(torch.bfloat16)
    attn = torch.einsum('benk,bemk->benm', q_heads, k_heads) * mask
    output = torch.einsum('benm,bmv->benv', attn.to(torch.bfloat16), v).to(torch.bfloat16)
    output = torch.sum(output, dim=1) # sum over projections
    kv_state = torch.einsum('benk,bnv->bekv', k_heads, v)
    return output.to(torch.bfloat16), kv_state.to(torch.bfloat16)

def save_tensors_to_file(tensor, filename, width=64):
    np_array = tensor.detach().to(torch.float32).cpu().numpy().reshape((-1,width))
    rep = '\n'.join([' '.join([f'{x:0.6f}' for x in row]) for row in np_array])
    print(f'Writing shape {np_array.shape} to {filename}')
    with open(filename, 'w') as f:
        f.write(rep)

def terraced_linear_attention_backward(grad_output, q, k, v, q_map, k_map):
    # Todo
    return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(q_map), torch.zeros_like(k_map)

def main():
    # Initialize mask
    mask = create_terraced_lower_triangular(N, 64)
    print('Generated mask')

    # Initialize tensors
    q, k, v, q_map, k_map = initialize_tensors()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    q_map.requires_grad_(True)
    k_map.requires_grad_(True)
    print('Initialized tensors')

    # Compute output (you'll implement this function)
    output, kv_state = compute_output(q, k, v, q_map, k_map, mask)
    print('Computed output')

    # Create a random gradient for the output
    o_grad = torch.randn_like(output)
    print('Generated random gradient for output')

    # Compute backward pass
    output.backward(o_grad)
    print('Computed backward pass')

    # Print gradients
    print('Gradient shapes:')
    print('q_grad:', q.grad.shape)
    print('k_grad:', k.grad.shape)
    print('v_grad:', v.grad.shape)
    print('q_map_grad:', q_map.grad.shape)
    print('k_map_grad:', k_map.grad.shape)

    # Compute grad the other way, with the same output
    grad_q, grad_k, grad_v, grad_q_map, grad_k_map = terraced_linear_attention_backward(o_grad, q, k, v, q_map, k_map)
    print('Computed backward pass')

    # Print some gradient statistics
    def print_grad_stats(name, grad):
        if grad is not None:
            print(f'{name} grad - min: {grad.min().item():.6f}, max: {grad.max().item():.6f}, mean: {grad.mean().item():.6f}, std: {grad.std().item():.6f}')
        else:
            print(f'{name} grad is None')

    print('\nGradient statistics:')
    print_grad_stats('grad_q_1', q.grad)
    print_grad_stats('grad_q_2', grad_q)
    print_grad_stats('grad_k_1', k.grad)
    print_grad_stats('grad_k_2', grad_k)
    print_grad_stats('grad_v_1', v.grad)
    print_grad_stats('grad_v_2', grad_v)
    print_grad_stats('grad_v_q_map_1', q_map.grad)
    print_grad_stats('grad_v_q_map_2', grad_q_map)
    print_grad_stats('grad_v_k_map_1', k_map.grad)
    print_grad_stats('grad_v_k_map_2', grad_k_map)

    # Save tensors to files
    save_tensors_to_file(q, 'q.txt', D_QK)
    save_tensors_to_file(k, 'k.txt', D_QK)
    save_tensors_to_file(v, 'v.txt', D_VO)
    save_tensors_to_file(q_map, 'q_map.txt', D_QK)
    save_tensors_to_file(k_map, 'k_map.txt', D_QK)
    save_tensors_to_file(output, 'reference_output.txt', D_VO)
    save_tensors_to_file(kv_state, 'reference_kv_state.txt', D_QK*D_VO)

    # print("Tensors have been initialized and saved to files.")
    # print("Output shape:", output.shape)

if __name__ == "__main__":
    main()