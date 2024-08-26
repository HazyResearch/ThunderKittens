import torch
import thunderkittens
from tqdm import tqdm

torch.manual_seed(42)

# Constants
B = 4       # Batch size, can't too be big, or the GPU memory will explode
H = 8       # Heads, which we're currently ignoring in this Python code.
N = 2048    # Sequence length
D_QK = 64   # Query/Key dimension
D_VO = 128  # Value/Output dimension
COLLABORATIVE_SMS = 8  # Number of collaborative SMs
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
    q = torch.randn(B, H, N, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    k = torch.randn(B, H, N, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    v = torch.randn(B, H, N, D_VO, dtype=torch.bfloat16, device='cuda:0')

    # Initialize Q_map and K_map tensors
    q_map = torch.randn(H, EXPANSION_FACTOR, D_QK, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK
    k_map = torch.randn(H, EXPANSION_FACTOR, D_QK, D_QK, dtype=torch.bfloat16, device='cuda:0') / D_QK

    return q, k, v, q_map, k_map

def compute_output(q, k, v, q_map, k_map, mask):
    q_heads = torch.relu(torch.einsum('hedk,bhnd->bhenk', q_map, q)).to(torch.bfloat16)
    k_heads = torch.relu(torch.einsum('hedk,bhnd->bhenk', k_map, k)).to(torch.bfloat16)
    attn = torch.einsum('bhenk,bhemk->bhenm', q_heads, k_heads) * mask
    output = torch.einsum('bhenm,bhmv->bhenv', attn.to(torch.bfloat16), v).to(torch.bfloat16)
    output = torch.sum(output, dim=2) # sum over projections
    kv_state = torch.einsum('bhenk,bhnv->bhekv', k_heads, v)
    return output.to(torch.bfloat16), kv_state.to(torch.float32)

def compare_tensors(name, tensor1, tensor2):
    diff = (tensor1 - tensor2).abs()
    avg_diff = diff.mean().item()
    max_diff = diff.max().item()
    rel_err = (avg_diff / (tensor2.abs().mean() + 1e-10)).item()
    if verbose:
        print(f"{name}:")
        print(f"  Avg diff: {avg_diff:.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  REF - min: {tensor2.min().item():.6f}, max: {tensor2.max().item():.6f}, mean: {tensor2.mean().item():.6f}, abs mean: {tensor2.abs().mean().item():.6f}, std: {tensor2.std().item():.6f}")
        print(f"  NEW - min: {tensor1.min().item():.6f}, max: {tensor1.max().item():.6f}, mean: {tensor1.mean().item():.6f}, abs mean: {tensor1.abs().mean().item():.6f}, std: {tensor1.std().item():.6f}")
        print(f"  Avg % error: {100*rel_err:.6f}%")
    if(rel_err > 1e-2):
        print(f"  Tensors differ significantly")
        exit(1)

def main():
    # Initialize mask
    mask = create_terraced_lower_triangular(N, 64)

    # Initialize tensors
    q, k, v, q_map, k_map = initialize_tensors()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    q_map.requires_grad_(True)
    k_map.requires_grad_(True)
    if verbose: print('Initialized tensors')

    # Compute output (you'll implement this function)
    output_ref, kv_state_ref = compute_output(q, k, v, q_map, k_map, mask)
    if verbose: print('Computed reference output')
    output, kv_state = torch.zeros_like(output_ref).to(torch.bfloat16).contiguous(), torch.zeros_like(kv_state_ref).to(torch.float32).contiguous()
    thunderkittens.cylon(q, k, v, output, kv_state, q_map, k_map)
    if verbose: print('Computed thunderkittens output')

    compare_tensors("Output", output, output_ref)
    compare_tensors("KV State", kv_state, kv_state_ref)

    # Create a random gradient for the output
    o_grad = torch.randn_like(output).to(torch.bfloat16)
    if verbose: print('Generated random gradient for output')

    # Compute backward pass
    output_ref.backward(o_grad)
    if verbose: print('Computed reference backward pass')

    # Compute grad the other way, with the same output
    grad_q, grad_k, grad_v, grad_q_map, grad_k_map = torch.zeros_like(q.grad, dtype=torch.float32), torch.zeros_like(k.grad, dtype=torch.float32), torch.zeros_like(v.grad, dtype=torch.float32), torch.zeros_like(q_map.grad, dtype=torch.float32), torch.zeros_like(k_map.grad, dtype=torch.float32)
    thunderkittens.cylon_bwd(q, k, v, q_map, k_map, o_grad, kv_state, grad_q, grad_k, grad_v, grad_q_map, grad_k_map)
    if verbose: print('Computed TK backward pass')

    compare_tensors("q_grad", grad_q, q.grad)
    compare_tensors("k_grad", grad_k, k.grad)
    compare_tensors("v_grad", grad_v, v.grad)
    compare_tensors("q_map_grad", grad_q_map, q_map.grad)
    compare_tensors("k_map_grad", grad_k_map, k_map.grad)

if __name__ == "__main__":
    verbose = True
    main()
    verbose = False
    for _ in tqdm(range(1000)):
        main()