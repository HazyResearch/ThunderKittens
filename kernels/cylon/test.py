import torch
import numpy as np

# Constants matching the C++ harness
LOAD_B = 1             # Original batch size used for generating inputs
B = 64                 # Actual batch size for testing
H = 1                  # Number of heads
N = 8192              # Sequence length
D_QK = 64             # Query/Key dimension
D_VO = 128            # Value/Output dimension
COLLABORATIVE_SMS = 8  # Number of collaborative SMs
STATE_PER_SM = 2      # How many maps get run on each SM
EXPANSION_FACTOR = COLLABORATIVE_SMS * 2

def load_reference_file(filename, expected_size):
    """Load and verify reference data from file"""
    try:
        data = np.loadtxt(filename)
        data = data.flatten()
        if len(data) != expected_size:
            print(f"Warning: {filename} contains {len(data)} elements, expected {expected_size}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def compare_outputs(name, computed, reference, expected_shape=None, tolerance=1e-3):
    """Compare computed outputs with reference values"""
    if isinstance(computed, torch.Tensor):
        # Convert to float32 before converting to numpy
        computed = computed.to(torch.float32).cpu().numpy()
    
    # Ensure reference shape matches computed shape
    if expected_shape is not None:
        reference = reference.reshape(expected_shape).flatten()
    
    # Flatten computed output for comparison
    computed_flat = computed.flatten()
    
    diff = np.abs(computed_flat - reference)
    rel_diff = diff / (np.abs(reference) + 1e-8)
    
    total_error = np.sum(diff)
    total_magnitude = np.sum(np.abs(reference))
    avg_error = total_error / len(reference)
    avg_magnitude = total_magnitude / len(reference)
    avg_percent_diff = (avg_error / avg_magnitude) * 100
    
    print(f"\n{name} comparison:")
    print(f"  Shape: computed {computed.shape}, reference {reference.shape}")
    print(f"  Max absolute difference: {np.max(diff):.6f}")
    print(f"  Mean absolute difference: {np.mean(diff):.6f}")
    print(f"  Max relative difference: {np.max(rel_diff):.6f}")
    print(f"  Mean relative difference: {np.mean(rel_diff):.6f}")
    print(f"  Average error: {avg_error:.6f}")
    print(f"  Average magnitude of reference: {avg_magnitude:.6f}")
    print(f"  Average percent difference: {avg_percent_diff:.6f}%")
    
    mismatches = {
        0.0001: np.sum(diff > 0.0001),
        0.001: np.sum(diff > 0.001),
        0.01: np.sum(diff > 0.01),
        0.1: np.sum(diff > 0.1)
    }
    
    for threshold, count in mismatches.items():
        print(f"  Mismatches > {threshold}: {count} ({100.0 * count / len(computed):.2f}%)")

def main():
    # Load reference inputs from files
    print("Loading reference data...")
    q_ref = load_reference_file("inputs/q.txt", LOAD_B * H * N * D_QK)
    k_ref = load_reference_file("inputs/k.txt", LOAD_B * H * N * D_QK)
    v_ref = load_reference_file("inputs/v.txt", LOAD_B * H * N * D_VO)
    q_map_ref = load_reference_file("inputs/q_map.txt", H * COLLABORATIVE_SMS * D_QK * D_QK * STATE_PER_SM)
    k_map_ref = load_reference_file("inputs/k_map.txt", H * COLLABORATIVE_SMS * D_QK * D_QK * STATE_PER_SM)
    o_grad_ref = load_reference_file("inputs/o_grad.txt", LOAD_B * H * N * D_VO)

    # Create input tensors
    q = torch.from_numpy(q_ref).to(torch.bfloat16).reshape(LOAD_B, H, N, D_QK).cuda()
    k = torch.from_numpy(k_ref).to(torch.bfloat16).reshape(LOAD_B, H, N, D_QK).cuda()
    v = torch.from_numpy(v_ref).to(torch.bfloat16).reshape(LOAD_B, H, N, D_VO).cuda()
    
    # For q_map and k_map layout:
    # From kernel:
    #   coord<st_bf<64, 64>> q_map_idx = {head_id, state_id/STATE_PER_SM, state_id%STATE_PER_SM, 0}
    # Total size must be H * COLLABORATIVE_SMS * D_QK * D_QK * STATE_PER_SM = 65536
    q_map = torch.from_numpy(q_map_ref).to(torch.bfloat16).reshape(H, COLLABORATIVE_SMS, D_QK * STATE_PER_SM, D_QK).cuda()
    k_map = torch.from_numpy(k_map_ref).to(torch.bfloat16).reshape(H, COLLABORATIVE_SMS, D_QK * STATE_PER_SM, D_QK).cuda()

    o_grad = torch.from_numpy(o_grad_ref).to(torch.bfloat16).reshape(LOAD_B, H, N, D_VO).cuda()

    print("Input tensor shapes:")
    print(f"q: {q.shape}")
    print(f"k: {k.shape}")
    print(f"v: {v.shape}")
    print(f"q_map: {q_map.shape}")
    print(f"k_map: {k_map.shape}")
    print(f"o_grad: {o_grad.shape}")

    # Run forward pass through thunderkittens kernel
    print("\nRunning forward pass...")
    
    import thunderkittens
    tk_output, tk_kv_state = thunderkittens.cylon(q, k, v, q_map, k_map)

    # Load reference outputs
    ref_output = load_reference_file("reference/output.txt", LOAD_B * H * N * D_VO)
    ref_kv_state = load_reference_file("reference/kv_state.txt", LOAD_B * H * STATE_PER_SM * COLLABORATIVE_SMS * D_QK * D_VO)
    
    # Compare forward pass outputs
    print("\nComparing forward pass outputs...")
    compare_outputs("Output", tk_output, ref_output, 
                   expected_shape=(LOAD_B, H, N, D_VO))
    compare_outputs("KV State", tk_kv_state, ref_kv_state, 
                   expected_shape=(LOAD_B, H, STATE_PER_SM * COLLABORATIVE_SMS * D_QK, D_VO))

    # Run backward pass
    print("\nRunning backward pass...")
    grad_q, grad_k, grad_v, grad_q_map, grad_k_map = thunderkittens.cylon_bwd(
        q, k, v, q_map, k_map, o_grad.to(torch.bfloat16), tk_kv_state.to(torch.float32)
    )

    # Load reference gradients
    ref_q_grad = load_reference_file("reference/q_grad.txt", LOAD_B * H * N * D_QK)
    ref_k_grad = load_reference_file("reference/k_grad.txt", LOAD_B * H * N * D_QK)
    ref_v_grad = load_reference_file("reference/v_grad.txt", LOAD_B * H * N * D_VO)
    ref_q_map_grad = load_reference_file("reference/q_map_grad.txt", H * COLLABORATIVE_SMS * D_QK * D_QK * STATE_PER_SM)
    ref_k_map_grad = load_reference_file("reference/k_map_grad.txt", H * COLLABORATIVE_SMS * D_QK * D_QK * STATE_PER_SM)

    # Compare backward pass outputs
    print("\nComparing backward pass outputs...")
    
    compare_outputs("Q Gradient", grad_q, ref_q_grad, 
                   expected_shape=(LOAD_B, H, N, D_QK))
    compare_outputs("K Gradient", grad_k, ref_k_grad, 
                   expected_shape=(LOAD_B, H, N, D_QK))
    compare_outputs("V Gradient", grad_v, ref_v_grad, 
                   expected_shape=(LOAD_B, H, N, D_VO))
    compare_outputs("Q Map Gradient", grad_q_map, ref_q_map_grad, 
                   expected_shape=(H, COLLABORATIVE_SMS, D_QK * STATE_PER_SM, D_QK))
    compare_outputs("K Map Gradient", grad_k_map, ref_k_map_grad,
                   expected_shape=(H, COLLABORATIVE_SMS, D_QK * STATE_PER_SM, D_QK))

if __name__ == "__main__":
    main()