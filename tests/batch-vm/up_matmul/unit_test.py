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

# Derived constants for cleaner indexing
K_TOTAL_DIM = NUM_K_TILES * K_TILE_SIZE
M_START = M_BLOCK_IDX * M_TILE_SIZE
M_END = (M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL) * M_TILE_SIZE
N_START = N_BLOCK_IDX * N_TILE_SIZE
N_END = (N_BLOCK_IDX + 1) * N_TILE_SIZE

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

def run_reference_implementation(rms_gate_intermediates, up_weights, gate_silu_intermediates, output_tensor_ref):
    """Runs the reference implementation for comparison."""
    # Extract the relevant blocks
    a_blocks = rms_gate_intermediates[M_START:M_END, :K_TOTAL_DIM]
    b_block = up_weights[N_START:N_END, :K_TOTAL_DIM]
    gate_silu_blocks = gate_silu_intermediates[M_START:M_END, N_START:N_END]

    # Split A blocks for processing
    a_block0 = a_blocks[:M_TILE_SIZE]
    a_block1 = a_blocks[M_TILE_SIZE:]
    gate_silu_block0 = gate_silu_blocks[:M_TILE_SIZE]
    gate_silu_block1 = gate_silu_blocks[M_TILE_SIZE:]
    
    # concatenate gate_silu_block0 and gate_silu_block1
    # gate_silu_block = torch.cat((gate_silu_block0, gate_silu_block1), dim=0)
    # return gate_silu_block0, gate_silu_block1

    # Process each block
    for i, (a_block, gate_silu_block) in enumerate([(a_block0, gate_silu_block0), (a_block1, gate_silu_block1)]):
        matmul = torch.matmul(a_block.float(), b_block.T.float())
        # Uncomment the following line when gate_silu multiplication is implemented
        output = (matmul * gate_silu_block.float()).to(output_tensor_ref.dtype)
        # output = matmul.to(output_tensor_ref.dtype)  # Currently just matmul without gate_silu
        output_tensor_ref[M_START + i*M_TILE_SIZE:M_START + (i+1)*M_TILE_SIZE, N_START:N_END] = output

    return output_tensor_ref

def main():
    print('\nGenerating inputs...')
    instructions, timings, barriers = generate_itb()
    rms_input_tensor, up_weights_tensor, gate_silu_intermediates, output_tensor_kernel = generate_tensor_inputs()

    print('Instruction shape:', instructions.shape)
    print('Barrier shape:', barriers.shape)
    print('Timings shape:', timings.shape)
    print('RMS Input (A) shape:', rms_input_tensor.shape)
    print('UP Weights (B) shape:', up_weights_tensor.shape)
    print('Gate SiLU Intermediates shape:', gate_silu_intermediates.shape)
    print('Output (C) shape:', output_tensor_kernel.shape)

    print('\nRunning the kernel...')
    kvm_llama(
        barriers,
        instructions,
        timings,
        up_weights_tensor,
        rms_input_tensor,
        gate_silu_intermediates,
        output_tensor_kernel
    )
    if TORCH_DEVICE.type == 'cuda':
        torch.cuda.synchronize(TORCH_DEVICE)

    print('\nRunning the reference implementation...')
    output_tensor_ref = torch.zeros_like(output_tensor_kernel)
    # reference_output_tile0, reference_output_tile1 = run_reference_implementation(
    #     rms_input_tensor, 
    #     up_weights_tensor, 
    #     gate_silu_intermediates, 
    #     output_tensor_ref
    # )
    output_tensor_ref = run_reference_implementation(
        rms_input_tensor, 
        up_weights_tensor, 
        gate_silu_intermediates, 
        output_tensor_ref
    )

    # Verify the output
    print('\nComparing outputs...')
    kernel_output_tile = output_tensor_kernel[M_START:M_END, N_START:N_END]
    reference_output_tile = output_tensor_ref[M_START:M_END, N_START:N_END]

    # Convert to float for comparison
    kernel_output_float = kernel_output_tile.float()
    reference_output_float = reference_output_tile.float()

    # split kernel output tile into two parts
    # reference_output_float0, reference_output_float1 = reference_output_tile0.float(), reference_output_tile1.float()

    # # split kernel output tile into two parts
    # kernel_output_tile0 = kernel_output_tile[:M_TILE_SIZE]
    # kernel_output_tile1 = kernel_output_tile[M_TILE_SIZE:]

    # print("Kernel output tile0:", kernel_output_tile0)
    # print("Kernel output tile1:", kernel_output_tile1)

    # #compare 
    # diff0 = torch.abs(kernel_output_tile0 - reference_output_float0)
    # max_diff0 = torch.max(diff0)
    # mean_diff0 = torch.mean(diff0)
    # diff1 = torch.abs(kernel_output_tile1 - reference_output_float1)
    # max_diff1 = torch.max(diff1)
    # mean_diff1 = torch.mean(diff1)

    # print(f"Max difference0: {max_diff0.item()}, Mean difference0: {mean_diff0.item()}")
    # print(f"Max difference1: {max_diff1.item()}, Mean difference1: {mean_diff1.item()}")
    
    # raise Exception("Stop here")

    

    print("Reference output tile:", reference_output_tile)
    # Calculate differences
    diff = torch.abs(kernel_output_float - reference_output_float)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    max_diff_idx = torch.where(diff == max_diff)

    print(f"Output tile (M={M_BLOCK_IDX} to {M_BLOCK_IDX + NUM_M_SUB_BLOCKS_PER_KERNEL_CALL - 1}, N={N_BLOCK_IDX}):")
    print(f"  Max absolute difference: {max_diff.item()}")
    print(f"  Mean absolute difference: {mean_diff.item()}")
    print(f"  Max difference location: {max_diff_idx}")

    if max_diff > 0:
        # Only print first location of max difference if there are differences
        row_idx = max_diff_idx[0][0].item()
        col_idx = max_diff_idx[1][0].item()
        print(f"  First max difference at: (row={row_idx}, col={col_idx})")
        print(f"  Kernel value at max diff: {kernel_output_float[row_idx, col_idx].item()}")
        print(f"  Reference value at max diff: {reference_output_float[row_idx, col_idx].item()}")
    else:
        print("  Outputs match exactly!")
    # print(f"  Kernel value at max diff: {kernel_output_float[max_diff_idx].item()}")
    # print(f"  Reference value at max diff: {reference_output_float[max_diff_idx].item()}")

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

if __name__ == "__main__":
    main()
