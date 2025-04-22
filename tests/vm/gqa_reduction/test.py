import math
import torch
import time # Keep for reference implementation timing if desired
# No need to explicitly import torch.cuda.Event, it's part of torch.cuda
from gqa_reduction import gqa_reduction

TORCH_DEVICE = torch.device('cuda:2') # CHANGE THIS TO YOUR GPU
torch.cuda.set_device(TORCH_DEVICE)

# Fixed VM parameters
NUM_BLOCKS_REDUCTION = 32 # Must match NUM_Q_HEADS for the reduction kernel
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_REDUCTION_OPCODE = 2

# Model/Attention Parameters (match kernel constants)
H_q = 32    # number of query heads (NUM_Q_HEADS in CUDA)
D_h = 64    # dimension of each head (HEAD_DIM in CUDA)

# Test Configuration
NUM_PARTIALS = 8     # Number of partial results to reduce (must be <= MAX_PARTIALS)
MAX_PARTIALS = 1024  # Max intermediate partials storage capacity (M_a)

# --- Helper Functions ---

def generate_inputs(num_q_heads: int, head_dim: int, num_partials: int, max_partials_storage: int):
    # torch.manual_seed(321)
    # torch.cuda.manual_seed_all(321) # For reproducibility on GPU
    torch.manual_seed(345)
    torch.cuda.manual_seed_all(345) # For reproducibility on GPU
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123) # For reproducibility on GPU

    # LSE Partials Input: Shape (H_q, M_a), dtype float32
    LSE_partials_in = torch.full((num_q_heads, max_partials_storage), -float('inf'), dtype=torch.float32, device=TORCH_DEVICE)
    lse_values = torch.randn(num_q_heads, num_partials, dtype=torch.float32, device=TORCH_DEVICE) * 2.0 # Scale variance
    LSE_partials_in[:, :num_partials] = lse_values

    # O Partials Input: Shape (H_q, M_a, D_h), dtype float32
    O_partials_in = torch.zeros((num_q_heads, max_partials_storage, head_dim), dtype=torch.float32, device=TORCH_DEVICE)
    o_values = torch.randn(num_q_heads, num_partials, head_dim, dtype=torch.bfloat16, device=TORCH_DEVICE).float()
    O_partials_in[:, :num_partials, :] = o_values

    # Final Output Tensor: Shape (H_q, D_h), dtype bfloat16
    O_final_out = torch.zeros(num_q_heads, head_dim, dtype=torch.bfloat16, device=TORCH_DEVICE)

    return LSE_partials_in, O_partials_in, O_final_out

def generate_instructions_and_timings_reduction(num_q_heads_to_run: int, num_partials_total: int):
    '''Generates instructions for the reduction kernel.'''
    instructions = []
    max_instructions = 1 # Only one reduction instruction per Q head block

    # Assign one reduction instruction per Q head block
    for q_idx in range(num_q_heads_to_run):
        inst = [GQA_REDUCTION_OPCODE, num_partials_total] + [0] * (INSTRUCTION_WIDTH - 2)
        instructions.append([inst])

    # Pad instructions if NUM_Q_HEADS < NUM_BLOCKS_REDUCTION
    while len(instructions) < NUM_BLOCKS_REDUCTION:
         instructions.append([[0] * INSTRUCTION_WIDTH for _ in range(max_instructions)]) # Add padding instruction lists

    instructions_tensor = torch.tensor(instructions, dtype=torch.int32).to(device=TORCH_DEVICE)
    timings_tensor = torch.zeros((NUM_BLOCKS_REDUCTION, max_instructions, TIMING_WIDTH), dtype=torch.int32).to(device=TORCH_DEVICE)

    return instructions_tensor, timings_tensor


def reference_attention_reduction(
    L_partials: torch.Tensor, # Shape: (H_q, M_a)
    O_partials: torch.Tensor, # Shape: (H_q, M_a, D_h)
    num_partials_to_reduce: int,
    num_q_heads: int,
    head_dim: int
) -> torch.Tensor:
    O_final_ref = torch.zeros(num_q_heads, head_dim, dtype=torch.float32, device=L_partials.device)

    for head_idx in range(num_q_heads):
        if (head_idx != 3):
            continue
        lses = L_partials[head_idx, :num_partials_to_reduce].float()
        outs = O_partials[head_idx, :num_partials_to_reduce, :].float()


        max_lse = torch.max(lses)
        adjusted_factors = torch.exp2(lses - max_lse)
        new_denominator = adjusted_factors.sum()
        print(lses)
        print(outs)
        print("Max LSE: ", max_lse)
        print("Denominator: ", new_denominator)

        reduced = (outs * adjusted_factors.unsqueeze(1)).sum(dim=0) / new_denominator
        O_final_ref[head_idx] = reduced
        print(O_final_ref[head_idx])
        break

    return O_final_ref

# --- Main Execution ---

print('\n--- Generating Synthetic Inputs ---')
LSE_partials_py, O_partials_py, O_final_py = generate_inputs(
    H_q, D_h, NUM_PARTIALS, MAX_PARTIALS
)
print(f"Input LSE Partials shape (Python): {LSE_partials_py.shape} ({LSE_partials_py.dtype})")
print(f"Input O Partials shape (Python): {O_partials_py.shape} ({O_partials_py.dtype})")
print(f"Output O Final shape (Python): {O_final_py.shape} ({O_final_py.dtype})")

print('\n--- Reshaping Tensors for Kernel ---')
LSE_partials_kernel = LSE_partials_py.unsqueeze(0).unsqueeze(0)
O_partials_kernel = O_partials_py.unsqueeze(0)
O_final_kernel = O_final_py.unsqueeze(0).unsqueeze(2)

print(f"Input LSE Partials shape (Kernel): {LSE_partials_kernel.shape}")
print(f"Input O Partials shape (Kernel): {O_partials_kernel.shape}")
print(f"Output O Final shape (Kernel): {O_final_kernel.shape}")

print('\n--- Generating Instructions & Timings Structures ---')
reduction_instructions, reduction_timings = generate_instructions_and_timings_reduction(
    H_q, NUM_PARTIALS
)
print(f"Reduction Instructions shape: {reduction_instructions.shape}")
print(f"Reduction Timings shape: {reduction_timings.shape}")

print('\n--- Running gqa_reduction Kernel & Measuring Time ---')
# <<< MODIFICATION START >>>

# Ensure kernel is loaded/warmed up (optional, but good practice)
gqa_reduction(
    reduction_instructions, reduction_timings,
    LSE_partials_kernel, O_partials_kernel, O_final_kernel
)
torch.cuda.synchronize(TORCH_DEVICE)

# Timing using CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

num_iters = 0
if num_iters > 0:
    start_event.record()
    for _ in range(num_iters):
        gqa_reduction(
            reduction_instructions, reduction_timings,
            LSE_partials_kernel, O_partials_kernel, O_final_kernel
        )
    end_event.record()
    torch.cuda.synchronize(TORCH_DEVICE)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    time_per_iter_us = (elapsed_time_ms * 1e3) / num_iters
    print(f'Time per iter (reduction): {time_per_iter_us:.2f} us')


# Squeeze the output tensor back to the standard Python shape for comparison
# (1, H_q, 1, D_h) -> (H_q, D_h)
O_final_squeezed = O_final_kernel.squeeze(2).squeeze(0)

# Convert final CUDA output (bf16) to float32 for comparison
O_final_cuda = O_final_squeezed.float()
print(f"Output O Final shape (from kernel, squeezed): {O_final_cuda.shape} ({O_final_cuda.dtype})")


print('\n--- Running Reference Python Reduction ---')
start_time_ref = time.time() # Keep using time.time() for CPU reference if needed
O_final_ref = reference_attention_reduction(
    LSE_partials_py, O_partials_py, NUM_PARTIALS, H_q, D_h
)
end_time_ref = time.time()
print(f"Reference implementation execution time: {end_time_ref - start_time_ref:.4f} seconds")
print(f"Reference O Final shape: {O_final_ref.shape} ({O_final_ref.dtype})")

# --- Verification ---
print('\n--- Comparing Outputs ---')

abs_error = torch.abs(O_final_cuda - O_final_ref)
epsilon = 1e-9
relative_error = abs_error / (torch.abs(O_final_ref) + epsilon)

max_abs_err = torch.max(abs_error).item()
mean_abs_err = torch.mean(abs_error).item()
max_rel_err = torch.max(relative_error).item()
mean_rel_err = torch.mean(relative_error).item()

print(f"Max Absolute Error: {max_abs_err:.6e}")
print(f"Mean Absolute Error: {mean_abs_err:.6e}")
print(f"Max Relative Error: {max_rel_err:.6e}")
print(f"Mean Relative Error: {mean_rel_err:.6e}")

# --- Tolerance Check ---
abs_tolerance = 1

# Optional: Print specific values for debugging
head_to_check = 3
print(f"\n--- Checking Head {head_to_check} ---")
for j in range(D_h):
    # Compare value at index (head_to_check, j) in both tensors
    print(f"Dim {j}: CUDA Output: {O_final_cuda[head_to_check, j].item():.6f} vs Reference: {O_final_ref[head_to_check, j].item():.6f}")

print(f"\nMax Abs Error: {max_abs_err:.6e} vs Tolerance: {abs_tolerance:.1e}")
print(f"Max Rel Error: {max_rel_err:.6e} vs Tolerance: {abs_tolerance:.1e}") # Note: Relative error tolerance might need different tuning


if max_abs_err < abs_tolerance:
    print(f"\n✅ Test Passed (Max Abs Error < {abs_tolerance:.1e})")
else:
    print(f"\n❌ Test Failed (Max Abs Error >= {abs_tolerance:.1e})")

    fail_idx = torch.argmax(abs_error)
    fail_head = (fail_idx // D_h).item()
    fail_dim = (fail_idx % D_h).item()
    print(f"  Max error occurred at Head {fail_head}, Dim {fail_dim}")
    print(f"  CUDA Output value: {O_final_cuda[fail_head, fail_dim].item():.6f}")
    print(f"  Reference value:   {O_final_ref[fail_head, fail_dim].item():.6f}")
    print(f"  Absolute error:    {abs_error[fail_head, fail_dim].item():.6e}")