import math
import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {TORCH_DEVICE}")

# Fixed parameters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
RMS_NORM_OPCODE = 1
NUM_OPS = 7

# Llama 8B Model Parameters
HIDDEN_DIM = 4096
BATCH_SIZE = 128

# Kernel parameters
LAYER_IDX = 0
BATCH_START_IDX = 0
BATCH_END_IDX = 24
QKV_BLOCK_IDX = 0
RMS_NORM_EPS = 1e-5

def generate_tensor_inputs():
    """
    Generates input tensors for the RMS normalization kernel.
    Returns:
        hidden_states: [BATCH_SIZE, HIDDEN_DIM] Input activations
        rms_weights: [HIDDEN_DIM] RMS normalization weights
        rms_output: [BATCH_SIZE, HIDDEN_DIM] Output buffer
    """
    torch.manual_seed(123)
    # Input activations
    hidden_states = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # RMS normalization weights
    rms_weights = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # Output buffer
    rms_output = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    
    return hidden_states, rms_weights, rms_output

def generate_itb():
    """
    Generates instructions, timings, and barriers.
    """
    instructions = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, 1, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, 48), dtype=torch.uint32, device=TORCH_DEVICE)

    # Fill instruction: [opcode, layer, qkv_block, batch_idx]
    instruction_values = [RMS_NORM_OPCODE, LAYER_IDX, BATCH_START_IDX, BATCH_END_IDX]
    for i, val in enumerate(instruction_values):
        instructions[0, 0, i] = val

    return instructions, timings, barriers

def compute_rms_norm_reference(hidden_states, rms_weights, output_tensor_ref):
    """
    Compute reference RMS normalization output using PyTorch.
    Matches the exact computation done in the CUDA kernel.
    """
    # Convert to float32 for better numerical precision
    hidden_states = hidden_states.to(torch.float32)
    rms_weights = rms_weights.to(torch.float32)
    
    # Compute sum of squares
    squared = hidden_states ** 2
    # Sum across hidden dimension (8192)
    sum_squared = torch.sum(squared, dim=-1, keepdim=True)
    # Divide by exactly 8192 as done in CUDA kernel
    variance = sum_squared / HIDDEN_DIM
    # Apply RMS normalization scale
    hidden_states = hidden_states * torch.rsqrt(variance + RMS_NORM_EPS)
    # Apply learned weights
    output_tensor_ref = (hidden_states * rms_weights).to(output_tensor_ref.dtype)
    
    return output_tensor_ref
    

def main():
    print('\nGenerating inputs...')
    instructions, timings, barriers = generate_itb()
    hidden_states, rms_weights, output_tensor_kernel = generate_tensor_inputs()

    print('Instruction shape:', instructions.shape)
    print('Barrier shape:', barriers.shape)
    print('Timings shape:', timings.shape)
    print('Hidden States shape:', hidden_states.shape)
    print('RMS Weights shape:', rms_weights.shape)
    print('Output shape:', output_tensor_kernel.shape)

    print('\nRunning the kernel...')
    #while True:
    torch.zero_(output_tensor_kernel)
    kvm_llama(
        barriers,
        instructions,
        timings,
        rms_weights,
        hidden_states,
        output_tensor_kernel,
        RMS_NORM_EPS
    )
    if TORCH_DEVICE.type == 'cuda':
        torch.cuda.synchronize(TORCH_DEVICE)

    print('\nRunning the reference implementation...')
    output_tensor_ref = torch.zeros_like(output_tensor_kernel)
    output_tensor_ref = compute_rms_norm_reference(hidden_states, rms_weights, output_tensor_ref)

    print(f"Output tensor kernel: {output_tensor_kernel[BATCH_START_IDX:BATCH_END_IDX]}")
    print("Shape of output tensor kernel: ", output_tensor_kernel.shape )
    print(f"Output tensor ref: {output_tensor_ref[BATCH_START_IDX:BATCH_END_IDX]}")
    print("Shape of output tensor ref: ", output_tensor_ref.shape)
    # Verify the output
    print('\nComparing outputs...')
    # Extract only the batch row we're processing
    kernel_output_row = output_tensor_kernel[BATCH_START_IDX:BATCH_END_IDX]
    reference_output_row = output_tensor_ref[BATCH_START_IDX:BATCH_END_IDX]


    print(f"Comparing batch row {BATCH_START_IDX}:{BATCH_END_IDX}:")

    diff = torch.abs(kernel_output_row.float() - reference_output_row.float())

    print(f"  Max absolute difference: {torch.max(diff).item()}")

    print(f"  Mean absolute difference: {torch.mean(diff).item()}")
    if torch.allclose(kernel_output_row.float(), reference_output_row.float(), atol=1e-2):
        print("  Test PASSED")
    else:
        print("  Test FAILED")

    # Check barrier update
    expected_barrier_val = 1
    actual_barrier_val = barriers[LAYER_IDX, RMS_NORM_OPCODE - 1, 0].item()
    print(f"\nBarrier g.Bar[{LAYER_IDX}, {RMS_NORM_OPCODE - 1}, 0]:")
    print(f"  Expected value after kernel: ~{expected_barrier_val} (if initially 0 and one kernel ran)")
    print(f"  Actual value after kernel: {actual_barrier_val}")
    if actual_barrier_val >= expected_barrier_val:
        print("  Barrier update seems consistent.")
    else:
        print("  Barrier update might be incorrect or dummy didn't update.")

if __name__ == "__main__":
    main()
