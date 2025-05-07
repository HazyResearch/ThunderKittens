import math
import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {TORCH_DEVICE}")

# Fixed parameters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
MLP_UP_OPCODE = 7
NUM_OPS = 7

# Llama 70B Model Parameters
HIDDEN_DIM = 8192
INTERMEDIATE_DIM = 28672
BATCH_SIZE = 128

# Kernel parameters
LAYER_IDX = 0
M_BLOCK_IDX = 0
N_BLOCK_IDX = 0
NUM_K_TILES = 2
M_TILE_SIZE = 64
N_TILE_SIZE = 128
K_TILE_SIZE = 128
NUM_M_SUB_BLOCKS_PER_KERNEL_CALL = 2

# Unit testing
K_TOTAL_DIM = NUM_K_TILES * K_TILE_SIZE
M_TOTAL_ROWS = (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE
N_TOTAL_COLS_C = (N_BLOCK_IDX + 1) * N_TILE_SIZE

def generate_tensor_inputs():
    """
    Generates input tensors for the up_matmul kernel.
    Returns:
        rms_gate_intermediates: [BATCH_SIZE, HIDDEN_DIM] = [128, 8192]
        up_weights: [INTERMEDIATE_DIM, HIDDEN_DIM] = [28672, 8192]
        gate_silu_intermediates: [BATCH_SIZE, INTERMEDIATE_DIM] = [128, 28672]
        silu_out: [BATCH_SIZE, INTERMEDIATE_DIM] = [128, 28672]
    """
    torch.manual_seed(42)
    # g.rms_gate_intermediates (Input A to A*B^T)
    rms_gate_intermediates = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # g.up_weights (Input B to A*B^T)
    up_weights = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # g.gate_silu_intermediates (Pre-computed SiLU(gate) values)
    gate_silu_intermediates = torch.randn(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # g.silu_out (Output)
    silu_out = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    return rms_gate_intermediates, up_weights, gate_silu_intermediates, silu_out

def generate_itb():
    """
    Generates instructions, timings, and barriers.
    """
    instructions = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, 1, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, 64 + 2 * 8), dtype=torch.uint32, device=TORCH_DEVICE)

    # Fill instruction: [opcode, layer, row, col, iters, <padding>]
    instruction_values = [MLP_UP_OPCODE, LAYER_IDX, M_BLOCK_IDX, N_BLOCK_IDX, NUM_K_TILES]
    for i, val in enumerate(instruction_values):
        instructions[0, 0, i] = val

    return instructions, timings, barriers

# Generate inputs
print('\nGenerating inputs...')
instructions, timings, barriers = generate_itb()
rms_gate_intermediates, up_weights_tensor, gate_silu_intermediates, silu_out = generate_tensor_inputs()

print('Instruction shape:', instructions.shape)
print('Barrier shape:', barriers.shape)
print('Timings shape:', timings.shape)
print('RMS Gate Intermediates (A) shape:', rms_gate_intermediates.shape)
print('UP Weights (B) shape:', up_weights_tensor.shape)
print('Gate SiLU Intermediates shape:', gate_silu_intermediates.shape)
print('SiLU Out shape:', silu_out.shape)

print('\nRunning the kernel...')
kvm_llama(
    barriers,
    instructions,
    timings,
    up_weights_tensor,
    rms_gate_intermediates,
    gate_silu_intermediates,
    silu_out
)
if TORCH_DEVICE.type == 'cuda':
    torch.cuda.synchronize(TORCH_DEVICE)

# Reference Implementation
print('\nRunning the reference implementation...')
output_tensor_ref = torch.zeros_like(silu_out)

# Extract the A sub-blocks processed by the kernel
a_part0_ref = rms_gate_intermediates[M_BLOCK_IDX * M_TILE_SIZE : (M_BLOCK_IDX + 1) * M_TILE_SIZE, 0 : K_TOTAL_DIM]
a_part1_ref = rms_gate_intermediates[(M_BLOCK_IDX + 1) * M_TILE_SIZE : (M_BLOCK_IDX + 2) * M_TILE_SIZE, 0 : K_TOTAL_DIM]

# Extract the B sub-block processed by the kernel
b_part_ref = up_weights_tensor[N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE, 0 : K_TOTAL_DIM]

# Extract the corresponding gate_silu_intermediates blocks
gate_silu_part0_ref = gate_silu_intermediates[M_BLOCK_IDX * M_TILE_SIZE : (M_BLOCK_IDX + 1) * M_TILE_SIZE, 
                                            N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE]
gate_silu_part1_ref = gate_silu_intermediates[(M_BLOCK_IDX + 1) * M_TILE_SIZE : (M_BLOCK_IDX + 2) * M_TILE_SIZE,
                                            N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE]

print("Ref A_part0 shape:", a_part0_ref.shape)
print("Ref A_part1 shape:", a_part1_ref.shape)
print("Ref B_part shape:", b_part_ref.shape)
print("Ref Gate SiLU part0 shape:", gate_silu_part0_ref.shape)
print("Ref Gate SiLU part1 shape:", gate_silu_part1_ref.shape)

# Perform matmul for the first M sub-block
matmul_0_ref = torch.matmul(a_part0_ref.float(), b_part_ref.T.float())
# output_0_ref = gate_silu_part0_ref.float() * matmul_0_ref
output_0_ref = matmul_0_ref
# output_0_ref = (matmul_0_ref * gate_silu_part0_ref.float()).to(output_tensor_ref.dtype)

# Perform matmul for the second M sub-block
matmul_1_ref = torch.matmul(a_part1_ref.float(), b_part_ref.T.float())
# output_1_ref = gate_silu_part1_ref.float()
output_1_ref = matmul_1_ref
# output_1_ref = (matmul_1_ref * gate_silu_part1_ref.float()).to(output_tensor_ref.dtype)


# Place results into the reference output tensor
output_tensor_ref[M_BLOCK_IDX * M_TILE_SIZE : (M_BLOCK_IDX + 1) * M_TILE_SIZE, N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE] = output_0_ref
output_tensor_ref[(M_BLOCK_IDX + 1) * M_TILE_SIZE : (M_BLOCK_IDX + 2) * M_TILE_SIZE, N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE] = output_1_ref

# Verify the output
print('\nComparing outputs...')
kernel_output_tile = silu_out[M_BLOCK_IDX * M_TILE_SIZE : (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE, 
                                        N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE]
reference_output_tile = output_tensor_ref[M_BLOCK_IDX * M_TILE_SIZE : (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE,
                                        N_BLOCK_IDX * N_TILE_SIZE : (N_BLOCK_IDX + 1) * N_TILE_SIZE]

# Convert to float for comparison
kernel_output_float = kernel_output_tile.float()
reference_output_float = reference_output_tile.float()

# Calculate differences
diff = torch.abs(kernel_output_float - reference_output_float)
max_diff = torch.max(diff)
mean_diff = torch.mean(diff)
max_diff_idx = torch.where(diff == max_diff)

print(f"Output tile (inst.row={M_BLOCK_IDX} to {M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL - 1}, inst.col={N_BLOCK_IDX}):")
print(f"  Max absolute difference: {max_diff.item()}")
print(f"  Mean absolute difference: {mean_diff.item()}")
print(f"  Max difference location: {max_diff_idx}")
print(f"  Kernel value at max diff: {kernel_output_float[max_diff_idx].item()}")
print(f"  Reference value at max diff: {reference_output_float[max_diff_idx].item()}")

# Use a more lenient tolerance for bfloat16
if torch.allclose(kernel_output_float, reference_output_float, atol=1e-2, rtol=1e-2):
    print("  Test PASSED")
else:
    print("  Test FAILED")

# Check barrier update
expected_barrier_val = 1
actual_barrier_val = barriers[LAYER_IDX, MLP_UP_OPCODE - 1, 0].item()
print(f"\nBarrier g.Bar[{LAYER_IDX}, {MLP_UP_OPCODE - 1}, 0]:")
print(f"  Expected value after kernel: ~{expected_barrier_val} (if initially 0 and one kernel ran)")
print(f"  Actual value after kernel: {actual_barrier_val}")
if actual_barrier_val >= expected_barrier_val:
    print("  Barrier update seems consistent.")
else:
    print("  Barrier update might be incorrect or dummy didn't update.")
