import torch
import numpy as np

torch.manual_seed(42)

# Constants
B = 1     # Batch size
H = 1     # Heads, which we're currently ignoring in this Python code.
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

def terraced_linear_attention_backward(kv_state, grad_output, q, k, v, q_map, k_map):
    q_grad, k_grad, v_grad = torch.zeros_like(q).to(torch.float32), torch.zeros_like(k).to(torch.float32), torch.zeros_like(v).to(torch.float32)
    q_map_grad, k_map_grad = torch.zeros_like(q_map).to(torch.float32), torch.zeros_like(k_map).to(torch.float32)

    # We need to iterate backwards through the blocks in chunks of 64
    for b in range(B):
        for s in range(EXPANSION_FACTOR):
            kv = -kv_state[b, s, :, :].to(torch.float32) # negate at first, so that we can do mma's onto it with + later. And we'll just invert everything at the end.
            kv_grad = torch.zeros_like(kv).to(torch.float32)
            for _id in range(N // 64):
                chunk_id = (N//64)-_id-1
                qm, km = -q_map[s].to(torch.float32), -k_map[s].to(torch.float32)
                q_chunk = q[b, chunk_id*64:(chunk_id+1)*64].to(torch.float32)
                k_chunk = k[b, chunk_id*64:(chunk_id+1)*64].to(torch.float32)
                v_chunk = v[b, chunk_id*64:(chunk_id+1)*64].to(torch.float32)
                o_grad_chunk = grad_output[b, chunk_id*64:(chunk_id+1)*64]
                # First we need to subtract (add, since negated) the KV contribution from that local state.
                k_head_prerelu = torch.einsum('dk,nd->nk', km, k_chunk)
                kv += torch.einsum('nk,nv->kv', torch.relu(-k_head_prerelu), v_chunk) # we now have the kv state as it was before the last iter forwards

                # next, we need to compute q_head pre-relu and its corresponding gradients
                q_head_prerelu = torch.einsum('dk,nd->nk', qm, q_chunk)
                q_head_chunk_grad = torch.einsum('nv,kv->nk', o_grad_chunk, kv) * (q_head_prerelu<0) # incorporate grad of relu
                q_grad[b, chunk_id*64:(chunk_id+1)*64] += torch.einsum('nk,dk->nd', q_head_chunk_grad, qm)
                q_map_grad[s] += torch.einsum('dn,nk->dk', q_chunk.T, q_head_chunk_grad)

                # next we accumulate the gradient on the current kv state
                kv_grad += torch.einsum('nk,nv->kv', (q_head_prerelu<0)*q_head_prerelu, o_grad_chunk)

                v_grad[b, chunk_id*64:(chunk_id+1)*64] += torch.einsum('kv,nk->nv', kv_grad, (k_head_prerelu<0)*k_head_prerelu)
                
                k_head_chunk_grad = torch.einsum('nv,kv->nk', v_chunk, kv_grad) * (k_head_prerelu<0)
                k_grad[b, chunk_id*64:(chunk_id+1)*64] += torch.einsum('nk,dk->nd', k_head_chunk_grad, km)
                k_map_grad[s] += torch.einsum('dn,nk->dk', k_chunk.T, k_head_chunk_grad)

                print(f'iter {_id} kv.abs().mean().item(): {kv.abs().mean().item()}')

    for s in range(EXPANSION_FACTOR):
        q_map_grad[s] = -q_map_grad[s]
        k_map_grad[s] = -k_map_grad[s]

    # Todo
    return q_grad, k_grad, v_grad, q_map_grad, k_map_grad

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
    o_grad = torch.randn_like(output).to(torch.float32)
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
    grad_q, grad_k, grad_v, grad_q_map, grad_k_map = terraced_linear_attention_backward(kv_state, o_grad, q, k, v, q_map, k_map)
    print('Computed backward pass')

    # Print some gradient statistics
    def print_grad_stats(name, grad, gref):
        print(f'{name} REF - min: {gref.min().item():.6f}, max: {gref.max().item():.6f}, mean: {gref.mean().item():.6f}, abs mean: {gref.abs().mean().item():.6f}, std: {gref.std().item():.6f}')
        print(f'{name} NEW - min: {grad.min().item():.6f}, max: {grad.max().item():.6f}, mean: {grad.mean().item():.6f}, abs mean: {grad.abs().mean().item():.6f}, std: {grad.std().item():.6f}')
        print(f'{name} avg % error: {100*((grad-gref).abs().mean()/gref.abs().mean()).item():.6f}%')

    print('\nGradient statistics:')
    print_grad_stats('grad_q', grad_q, q.grad)
    print_grad_stats('grad_k', grad_k, k.grad)
    print_grad_stats('grad_v', grad_v, v.grad)
    print_grad_stats('grad_q_map', grad_q_map, q_map.grad)
    print_grad_stats('grad_k_map', grad_k_map, k_map.grad)

    print(grad_k[0,0,:10])
    print(k.grad[0,0,:10])

    # Save tensors to files
    save_tensors_to_file(q, 'q.txt', D_QK)
    save_tensors_to_file(k, 'k.txt', D_QK)
    save_tensors_to_file(v, 'v.txt', D_VO)
    save_tensors_to_file(q_map, 'q_map.txt', D_QK)
    save_tensors_to_file(k_map, 'k_map.txt', D_QK)
    save_tensors_to_file(output, 'reference_output.txt', D_VO)
    save_tensors_to_file(kv_state, 'reference_kv_state.txt', D_QK*D_VO)

    print("Tensors have been initialized and saved to files.")
    print("Output shape:", output.shape)
    print("KV state shape:", kv_state.shape)

if __name__ == "__main__":
    main()