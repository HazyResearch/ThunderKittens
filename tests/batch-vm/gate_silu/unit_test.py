import math
import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {TORCH_DEVICE}")

# Fixed parameters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
MLP_GATE_OPCODE = 6
NUM_OPS = 7

# Llama 70B Model Parameters
HIDDEN_DIM = 8192
INTERMEDIATE_DIM = 28672
BATCH_SIZE = 128

# Kernel parameters
LAYER_IDX = 0
M_BLOCK_IDX = 0
N_BLOCK_IDX = 0
NUM_K_TILES = HIDDEN_DIM // 128
M_TILE_SIZE = 64
N_TILE_SIZE = 128
K_TILE_SIZE = 128
NUM_M_SUB_BLOCKS_PER_KERNEL_CALL = 2

# Derived constants for cleaner indexing
K_TOTAL_DIM = NUM_K_TILES * K_TILE_SIZE
M_START = M_BLOCK_IDX * M_TILE_SIZE
M_END = (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE
N_START = N_BLOCK_IDX * N_TILE_SIZE
N_END = (N_BLOCK_IDX + 1) * N_TILE_SIZE

def generate_tensor_inputs():
    """
    Generates input tensors for the gate_silu kernel.
    Returns:
        rms_input: [BATCH_SIZE, HIDDEN_DIM]
        gate_weights: [INTERMEDIATE_DIM, HIDDEN_DIM]
        output_silu: [BATCH_SIZE, INTERMEDIATE_DIM]
    """
    torch.manual_seed(42)
    # g.rms_gate_intermediates (Input A to A*B^T)
    rms_input = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # g.gate_weights (Input B to A*B^T)
    gate_weights = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # g.gate_silu_intermediates (Output C)
    output_silu = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    return rms_input, gate_weights, output_silu

def generate_itb():
    """
    Generates instructions, timings, and barriers.
    """
    instructions = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, 1, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, 64 + 2 * 8), dtype=torch.uint32, device=TORCH_DEVICE)

    # Fill instruction: [opcode, layer, row, col, iters, <padding>]
    instruction_values = [MLP_GATE_OPCODE, LAYER_IDX, M_BLOCK_IDX, N_BLOCK_IDX, NUM_K_TILES]
    for i, val in enumerate(instruction_values):
        instructions[0, 0, i] = val

    return instructions, timings, barriers

def run_reference_implementation(rms_input, gate_weights, output_tensor_ref):
    """Runs the reference implementation for comparison."""
    # Extract the relevant blocks
    a_blocks = rms_input[M_START:M_END, :K_TOTAL_DIM]
    b_block = gate_weights[N_START:N_END, :K_TOTAL_DIM]

    # Split A blocks for processing
    a_block0 = a_blocks[:M_TILE_SIZE]
    a_block1 = a_blocks[M_TILE_SIZE:]

    # Process each block
    for i, a_block in enumerate([a_block0, a_block1]):
        matmul = torch.matmul(a_block.float(), b_block.T.float())
        silu = (matmul * torch.sigmoid(matmul)).to(output_tensor_ref.dtype)
        output_tensor_ref[M_START + i*M_TILE_SIZE:M_START + (i+1)*M_TILE_SIZE, N_START:N_END] = silu

    return output_tensor_ref

def main():
    print('\nGenerating inputs...')
    instructions, timings, barriers = generate_itb()
    rms_input_tensor, gate_weights_tensor, output_tensor_kernel = generate_tensor_inputs()

    print('Instruction shape:', instructions.shape)
    print('Barrier shape:', barriers.shape)
    print('Timings shape:', timings.shape)
    print('RMS Input (A) shape:', rms_input_tensor.shape)
    print('Gate Weights (B) shape:', gate_weights_tensor.shape)
    print('Output SiLU (C) shape:', output_tensor_kernel.shape)

    print('\nRunning the kernel...')
    kvm_llama(
        barriers,
        instructions,
        timings,
        gate_weights_tensor,
        rms_input_tensor,
        output_tensor_kernel
    )
    if TORCH_DEVICE.type == 'cuda':
        torch.cuda.synchronize(TORCH_DEVICE)

    print('\nRunning the reference implementation...')
    output_tensor_ref = torch.zeros_like(output_tensor_kernel)
    output_tensor_ref = run_reference_implementation(rms_input_tensor, gate_weights_tensor, output_tensor_ref)

    # Verify the output
    print('\nComparing outputs...')
    kernel_output_tile = output_tensor_kernel[M_START:M_END, N_START:N_END]
    reference_output_tile = output_tensor_ref[M_START:M_END, N_START:N_END]

    diff = torch.abs(kernel_output_tile.float() - reference_output_tile.float())
    print(f"Output tile (M={M_BLOCK_IDX} to {M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL - 1}, N={N_BLOCK_IDX}):")
    print(f"  Max absolute difference: {torch.max(diff).item()}")
    print(f"  Mean absolute difference: {torch.mean(diff).item()}")
    if torch.allclose(kernel_output_tile.float(), reference_output_tile.float(), atol=1e-3):
        print("  Test PASSED")
    else:
        print("  Test FAILED")

    # Check barrier update
    expected_barrier_val = 1
    actual_barrier_val = barriers[LAYER_IDX, MLP_GATE_OPCODE - 1, 0].item()
    print(f"\nBarrier g.Bar[{LAYER_IDX}, {MLP_GATE_OPCODE - 1}, 0]:")
    print(f"  Expected value after kernel: ~{expected_barrier_val} (if initially 0 and one kernel ran)")
    print(f"  Actual value after kernel: {actual_barrier_val}")
    if actual_barrier_val >= expected_barrier_val:
        print("  Barrier update seems consistent.")
    else:
        print("  Barrier update might be incorrect or dummy didn't update.")

if __name__ == "__main__":
    main()