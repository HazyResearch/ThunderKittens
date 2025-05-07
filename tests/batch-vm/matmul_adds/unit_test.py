import math
import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {TORCH_DEVICE}")

# Fixed parameters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
O_PROJ_OPCODE = 4
NUM_OPS = 7

# Llama 70B Model Parameters
HIDDEN_DIM = 8192
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

# Derived constants for cleaner indexing
K_TOTAL_DIM = NUM_K_TILES * K_TILE_SIZE
M_START = M_BLOCK_IDX * M_TILE_SIZE
M_END = (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE
N_START = N_BLOCK_IDX * N_TILE_SIZE
N_END = (N_BLOCK_IDX + 1) * N_TILE_SIZE

def generate_tensor_inputs():
    """
    Generates input tensors for the o_proj kernel.
    Returns:
        attn_out: [BATCH_SIZE, HIDDEN_DIM]
        o_weights: [HIDDEN_DIM, HIDDEN_DIM]
        hidden_states: [BATCH_SIZE, HIDDEN_DIM]
    """
    torch.manual_seed(42)
    # attn_out (Input A to A*B^T)
    attn_out = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # o_weights (Input B to A*B^T)
    o_weights = torch.randn(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # hidden_states (Output C)
    hidden_states = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    return attn_out, o_weights, hidden_states

def generate_itb():
    """
    Generates instructions, timings, and barriers.
    """
    instructions = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, 1, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, 64 + 2 * 8), dtype=torch.uint32, device=TORCH_DEVICE)

    # Fill instruction: [opcode, layer, row, col, iters, <padding>]
    instruction_values = [O_PROJ_OPCODE, LAYER_IDX, M_BLOCK_IDX, N_BLOCK_IDX, NUM_K_TILES]
    for i, val in enumerate(instruction_values):
        instructions[0, 0, i] = val

    return instructions, timings, barriers

def run_reference_implementation(attn_out, o_weights, output_tensor_ref):
    """Runs the reference implementation for comparison."""
    # Extract the relevant blocks
    a_blocks = attn_out[M_START:M_END, :K_TOTAL_DIM]
    b_block = o_weights[N_START:N_END, :K_TOTAL_DIM]

    # Split A blocks for processing
    a_block0 = a_blocks[:M_TILE_SIZE]
    a_block1 = a_blocks[M_TILE_SIZE:]

    # Process each block
    for i, a_block in enumerate([a_block0, a_block1]):
        matmul = torch.matmul(a_block.float(), b_block.T.float())
        # Add the result to existing values in output_tensor_ref
        output_tensor_ref[M_START + i*M_TILE_SIZE:M_START + (i+1)*M_TILE_SIZE, N_START:N_END] += matmul.to(output_tensor_ref.dtype)

    return output_tensor_ref

def main():
    print('\nGenerating inputs...')
    instructions, timings, barriers = generate_itb()
    attn_out_tensor, o_weights_tensor, hidden_states_tensor = generate_tensor_inputs()

    print('Instruction shape:', instructions.shape)
    print('Barrier shape:', barriers.shape)
    print('Timings shape:', timings.shape)
    print('Attention Output (A) shape:', attn_out_tensor.shape)
    print('O Weights (B) shape:', o_weights_tensor.shape)
    print('Hidden States (C) shape:', hidden_states_tensor.shape)

    # --- BEGIN SLICE PRINTING ---
    print('\n--- Slices of Input Tensors Used by Kernel ---')

    # Define how many elements to show, e.g., top-left 4x4 corner of the relevant block
    slice_preview_rows = 4
    slice_preview_cols = 4

    # Relevant block of attn_out_tensor (Input A)
    attn_out_block_used_by_kernel = attn_out_tensor[M_START:M_END, :K_TOTAL_DIM]
    print(f"\nattn_out_tensor (Input A) block used by kernel (shape: {attn_out_block_used_by_kernel.shape}):")
    print(f"Top-left {slice_preview_rows}x{slice_preview_cols} slice of this block:")
    print(attn_out_block_used_by_kernel[:slice_preview_rows, :slice_preview_cols])

    # Relevant block of o_weights_tensor (Input B)
    o_weights_block_used_by_kernel = o_weights_tensor[N_START:N_END, :K_TOTAL_DIM]
    print(f"\no_weights_tensor (Input B) block used by kernel (shape: {o_weights_block_used_by_kernel.shape}):")
    print(f"Top-left {slice_preview_rows}x{slice_preview_cols} slice of this block:")
    print(o_weights_block_used_by_kernel[:slice_preview_rows, :slice_preview_cols])
    print('--- End of Slices ---')
    # --- END SLICE PRINTING ---

    print('\nRunning the kernel...')
    kvm_llama(
        barriers,
        instructions,
        timings,
        o_weights_tensor,
        attn_out_tensor,
        hidden_states_tensor
    )
    if TORCH_DEVICE.type == 'cuda':
        torch.cuda.synchronize(TORCH_DEVICE)

    print('\nRunning the reference implementation...')
    # If testing store_async, the reference should calculate matmul into an initially zeroed output
    output_tensor_ref = torch.zeros_like(hidden_states_tensor)
    output_tensor_ref = run_reference_implementation(attn_out_tensor, o_weights_tensor, output_tensor_ref)

    # Verify the output
    print('\nComparing outputs...')
    kernel_output_tile = hidden_states_tensor[M_START:M_END, N_START:N_END]
    reference_output_tile = output_tensor_ref[M_START:M_END, N_START:N_END]

    # --- BEGIN OUTPUT SLICE PRINTING ---
    print(f"\nKernel output tile (shape: {kernel_output_tile.shape}):")
    print(f"Top-left {slice_preview_rows}x{slice_preview_cols} slice:")
    print(kernel_output_tile[:slice_preview_rows, :slice_preview_cols])

    print(f"\nReference output tile (shape: {reference_output_tile.shape}):")
    print(f"Top-left {slice_preview_rows}x{slice_preview_cols} slice:")
    print(reference_output_tile[:slice_preview_rows, :slice_preview_cols])
    # --- END OUTPUT SLICE PRINTING ---

    diff = torch.abs(kernel_output_tile.float() - reference_output_tile.float())
    print(f"\nOutput tile (M={M_BLOCK_IDX} to {M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL - 1}, N={N_BLOCK_IDX}):")
    print(f"  Max absolute difference: {torch.max(diff).item()}")
    print(f"  Mean absolute difference: {torch.mean(diff).item()}")
    # Consider adjusting atol for bfloat16 comparisons
    if torch.allclose(kernel_output_tile.float(), reference_output_tile.float(), atol=1e-2, rtol=1e-2): # Adjusted tolerances
        print("  Test PASSED")
    else:
        print("  Test FAILED")
        # print("Kernel output tile values:")
        # print(kernel_output_tile)
        # print("Reference output tile values:")
        # print(reference_output_tile)
        # print("Difference:")
        # print(diff)


    # Check barrier update
    expected_barrier_val = 1
    actual_barrier_val = barriers[LAYER_IDX, O_PROJ_OPCODE - 1, 0].item()
    print(f"\nBarrier g.Bar[{LAYER_IDX}, {O_PROJ_OPCODE - 1}, 0]:")
    print(f"  Expected value after kernel: ~{expected_barrier_val} (if initially 0 and one kernel ran)")
    print(f"  Actual value after kernel: {actual_barrier_val}")
    if actual_barrier_val >= expected_barrier_val: # Check for >= in case other things increment the barrier
        print("  Barrier update seems consistent.")
    else:
        print("  Barrier update might be incorrect or dummy didn't update.")

if __name__ == "__main__":
    main()
