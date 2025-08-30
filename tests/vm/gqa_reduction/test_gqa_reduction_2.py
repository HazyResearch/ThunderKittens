import math
import torch
import time
# Ensure you import the kernel with the barrier argument
from gqa_reduction import gqa_reduction

TORCH_DEVICE = torch.device('cuda:6') # CHANGE THIS TO YOUR GPU
torch.cuda.set_device(TORCH_DEVICE)

# Fixed VM parameters
NUM_BLOCKS_REDUCTION = 32 // 4 # Must match NUM_Q_HEADS for the reduction kernel
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128

NUM_LAYERS = 16
NUM_OPS = 6
LAYER_IDX = 5 # Layer index to run the test on

# Opcodes
GQA_PARTIAL_OPCODE = 1 # Needed for barrier indexing convention
GQA_REDUCTION_OPCODE = 2

# Model/Attention Parameters (match kernel constants)
H_q = 32    # number of query heads (NUM_Q_HEADS in CUDA)
H_kv = 8    # number of KV heads (Needed for barrier tensor dimension)
D_h = 64    # dimension of each head (HEAD_DIM in CUDA)

# Test Configuration
NUM_PARTIALS = 8     # Number of partial results to reduce
MAX_PARTIALS = 1024  # Max intermediate partials storage capacity

# --- Helper Functions ---

def generate_inputs(num_q_heads: int, head_dim: int, num_partials: int, max_partials_storage: int):
    torch.manual_seed(345)
    torch.cuda.manual_seed_all(345)

    LSE_partials_in = torch.full((num_q_heads, max_partials_storage), -float('inf'), dtype=torch.float32, device=TORCH_DEVICE)
    lse_values = torch.randn(num_q_heads, num_partials, dtype=torch.float32, device=TORCH_DEVICE) * 2.0 # Scale variance
    LSE_partials_in[:, :num_partials] = lse_values

    O_partials_in = torch.zeros((num_q_heads, max_partials_storage, head_dim), dtype=torch.float32, device=TORCH_DEVICE)
    o_values = torch.randn(num_q_heads, num_partials, head_dim, dtype=torch.bfloat16, device=TORCH_DEVICE).float()
    O_partials_in[:, :num_partials, :] = o_values

    O_final_out = torch.zeros(num_q_heads, head_dim, dtype=torch.bfloat16, device=TORCH_DEVICE)

    return LSE_partials_in, O_partials_in, O_final_out

def generate_instructions_timings_barriers_reduction(
    num_q_heads_to_run: int,
    num_partials_total: int,
    layer_idx_to_run: int,
    num_layers: int,
    num_ops_barrier: int,
    num_kv_heads: int
):
    '''Generates instructions, timings, and initialized barriers for the reduction kernel.'''
    instructions = []
    max_instructions = 1 # Only one reduction instruction per Q head block

    # Assign one reduction instruction per Q head block
    for q_idx in range(num_q_heads_to_run // 4):
        # Instruction format: [OPCODE, LAYER_IDX, NUM_PARTIALS]
        inst = [GQA_REDUCTION_OPCODE, layer_idx_to_run, num_partials_total] + [0] * (INSTRUCTION_WIDTH - 3)
        instructions.append([inst]) # Each block gets a list containing one instruction

    # Pad instruction list if NUM_Q_HEADS < NUM_BLOCKS_REDUCTION (number of CUDA blocks)
    while len(instructions) < NUM_BLOCKS_REDUCTION:
         instructions.append([[0] * INSTRUCTION_WIDTH for _ in range(max_instructions)])

    instructions_tensor = torch.tensor(instructions, dtype=torch.int32).to(device=TORCH_DEVICE)

    # --- Initialize Barriers ---
    barriers = torch.zeros((num_layers, num_ops_barrier, num_q_heads_to_run + 2 * num_kv_heads), dtype=torch.int32).to(device=TORCH_DEVICE)

    # Simulate completion of the previous partial attention step (Opcode 1)
    # The reduction kernel waits for the counter at index GQA_REDUCTION_OPCODE - 1 (=1)
    # to reach num_partials_total for each Q head.
    barriers[layer_idx_to_run, GQA_REDUCTION_OPCODE - 1, :num_q_heads_to_run] = 1

    timings_tensor = torch.zeros((NUM_BLOCKS_REDUCTION, max_instructions, TIMING_WIDTH), dtype=torch.int32).to(device=TORCH_DEVICE)

    return instructions_tensor, timings_tensor, barriers

def reference_attention_reduction(
    L_partials: torch.Tensor, # Shape: (H_q, M_a)
    O_partials: torch.Tensor, # Shape: (H_q, M_a, D_h)
    num_partials_to_reduce: int,
    num_q_heads: int,
    head_dim: int
) -> torch.Tensor:
    # (Reference implementation - unchanged)
    O_final_ref = torch.zeros(num_q_heads, head_dim, dtype=torch.bfloat16, device=L_partials.device)
    for head_idx in range(num_q_heads):
        lses = L_partials[head_idx, :num_partials_to_reduce].float()
        outs = O_partials[head_idx, :num_partials_to_reduce, :].float()
        max_lse = torch.max(lses)
        adjusted_factors = torch.exp2(lses - max_lse)
        new_denominator = adjusted_factors.sum()
        if new_denominator.item() == 0:
             reduced = torch.zeros_like(outs[0])
        else:
            reduced = (outs * adjusted_factors.unsqueeze(1)).sum(dim=0) / new_denominator
        O_final_ref[head_idx] = reduced.to(torch.bfloat16) # Ensure output matches kernel type
    return O_final_ref

# --- Main Execution ---

print('\n--- Generating Synthetic Inputs ---')
LSE_partials_py, O_partials_py, O_final_py = generate_inputs(
    H_q, D_h, NUM_PARTIALS, MAX_PARTIALS
)
# (Print input shapes - unchanged)
print(f"Input LSE Partials shape (Python): {LSE_partials_py.shape} ({LSE_partials_py.dtype})")
print(f"Input O Partials shape (Python): {O_partials_py.shape} ({O_partials_py.dtype})")
print(f"Output O Final shape (Python): {O_final_py.shape} ({O_final_py.dtype})")


print('\n--- Reshaping Tensors for Kernel ---')
# (Tensor reshaping for kernel - unchanged)
LSE_partials_kernel = LSE_partials_py.unsqueeze(0).unsqueeze(0)
O_partials_kernel = O_partials_py.unsqueeze(0)
O_final_kernel = O_final_py.unsqueeze(0).unsqueeze(2)
print(f"Input LSE Partials shape (Kernel): {LSE_partials_kernel.shape}")
print(f"Input O Partials shape (Kernel): {O_partials_kernel.shape}")
print(f"Output O Final shape (Kernel): {O_final_kernel.shape}")


print('\n--- Generating Instructions, Timings & Barriers ---')
reduction_instructions, reduction_timings, reduction_barriers = generate_instructions_timings_barriers_reduction(
    H_q, NUM_PARTIALS, LAYER_IDX, NUM_LAYERS, NUM_OPS, H_kv
)

print(f"Reduction Instructions shape: {reduction_instructions.shape}")
print(f"Reduction Timings shape: {reduction_timings.shape}")
print(f"Reduction Barriers shape: {reduction_barriers.shape}")
print(f"Initial Barriers[LAYER {LAYER_IDX}, Op {GQA_REDUCTION_OPCODE - 1}]: {reduction_barriers[LAYER_IDX, GQA_REDUCTION_OPCODE - 1, :H_q]}")
print(f"Initial Barriers[LAYER {LAYER_IDX}, Op {GQA_REDUCTION_OPCODE}]: {reduction_barriers[LAYER_IDX, GQA_REDUCTION_OPCODE, :H_q]}")


print('\n--- Running gqa_reduction Kernel & Measuring Time ---')

# Create a copy of barriers to pass to the kernel, so we can check the original later if needed
barriers_for_kernel = reduction_barriers.clone()

# Warmup run
gqa_reduction(
    reduction_instructions,
    barriers_for_kernel,
    reduction_timings,
    LSE_partials_kernel, O_partials_kernel, O_final_kernel
)
torch.cuda.synchronize(TORCH_DEVICE)

# Timing runs
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
num_iters = 50
if num_iters > 0:
    # Reset barriers before timed runs if the kernel modifies them in place
    barriers_for_kernel = reduction_barriers.clone()
    start_event.record()
    for _ in range(num_iters):
        # Note: If kernel *doesn't* modify barriers in place, you might not need to clone each time.
        # But cloning ensures each timed iteration starts with the correct initial state.
        gqa_reduction(
            reduction_instructions,
            barriers_for_kernel, # ****** PASS BARRIERS ******
            reduction_timings,
            LSE_partials_kernel, O_partials_kernel, O_final_kernel
        )
    end_event.record()
    torch.cuda.synchronize(TORCH_DEVICE)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    time_per_iter_us = (elapsed_time_ms * 1e3) / num_iters
    print(f'Time per iter (reduction): {time_per_iter_us:.2f} us')

# <<< Barrier State Check >>>
print('\n--- Checking Barrier State After Kernel Execution ---')
final_barrier_values = barriers_for_kernel[LAYER_IDX, GQA_REDUCTION_OPCODE, :H_q]
expected_barrier_value = 1 # Each Q head block signals once

print(f"Final Barriers[LAYER {LAYER_IDX}, Op {GQA_REDUCTION_OPCODE}]: {final_barrier_values}")
if torch.all(final_barrier_values == expected_barrier_value):
    print("✅ Barrier state for reduction completion looks correct.")
else:
    print(f"❌ Barrier state check failed! Expected {expected_barrier_value}, got {final_barrier_values}")

# (Squeeze output tensor - unchanged)
O_final_squeezed = O_final_kernel.squeeze(2).squeeze(0)
O_final_cuda = O_final_squeezed.float()
print(f"Output O Final shape (from kernel, squeezed): {O_final_cuda.shape} ({O_final_cuda.dtype})")

print('\n--- Running Reference Python Reduction ---')
# (Reference run - unchanged)
O_final_ref = reference_attention_reduction(
    LSE_partials_py, O_partials_py, NUM_PARTIALS, H_q, D_h
)
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
abs_tolerance = 1e-2
# head_to_check = 3
# print(f"\n--- Checking Head {head_to_check} ---")
# for j in range(D_h):
#     print(f"Dim {j}: CUDA Output: {O_final_cuda[head_to_check, j].item():.6f} vs Reference: {O_final_ref[head_to_check, j].item():.6f}")
# print(f"\nMax Abs Error: {max_abs_err:.6e} vs Tolerance: {abs_tolerance:.1e}")
# print(f"Max Rel Error: {max_rel_err:.6e} vs Tolerance: {abs_tolerance:.1e}")


if max_abs_err < abs_tolerance:
    print(f"\n✅ Test Passed (Max Abs Error < {abs_tolerance:.1e})")
else:
    print(f"\n❌ Test Failed (Max Abs Error >= {abs_tolerance:.1e})")
    # (Error location printing - unchanged)
    # ...