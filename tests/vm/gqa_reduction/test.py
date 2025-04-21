import math
import torch
from gqa_partial import gqa_partial
from gqa_reduction import gqa_reduction # Import the new kernel binding

TORCH_DEVICE = torch.device('cuda:7') # Or your desired CUDA device

# --- Constants ---
# Fixed parameters
NUM_BLOCKS = 148 # VM size, must be >= H_q for reduction
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_PARTIAL_OPCODE = 1
GQA_REDUCTION_OPCODE = 2

# Llama 3.2 1B Model Parameters
L = 16 # number of hidden layers
B = 1 # this is fixed for now
N_max = 131072 # maximum sequence length
H_q = 32 # number of query heads
H_kv = 8 # number of key/value heads (number of query groups)
D_h = 64 # dimension of each head

# Kernel parameters
LAYER_IDX = 3
POS_ID = 1050
MAX_PARTIALS = 1024 # Max intermediates allowed by tensor shapes
ATTN_SCALE = 1 / math.sqrt(D_h)
NUM_PARTIALS = 2
ATTN_BLOCK_SIZE = 16

# --- Helper Functions ---

def generate_tensor_inputs(L: int, M_a: int, N_max: int, H_q: int, H_kv: int, D_h: int):
    '''Generate tensor inputs for the GQA kernels.'''
    torch.manual_seed(42)
    # Inputs for gqa_partial
    Q_in   = torch.randn(H_q, D_h,            dtype=torch.bfloat16, device=TORCH_DEVICE)
    K_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # Intermediates (Outputs of partial, Inputs to reduction)
    # Match kernel's expected layout for inputs (add batch dim 1)
    LSE_partials = torch.zeros(1, H_q, M_a,    dtype=torch.float32,  device=TORCH_DEVICE)
    O_partials   = torch.zeros(1, H_q, M_a, D_h, dtype=torch.float32,  device=TORCH_DEVICE)
    # Final Output (Output of reduction)
    # Match kernel's expected layout (add batch dims 1, 1) and type bf16
    O_final      = torch.zeros(1, H_q, 1, D_h,    dtype=torch.bfloat16, device=TORCH_DEVICE)

    print("\nInitial Tensor Shapes:")
    print(f"  Q_in (used by partial): {Q_in.shape}")
    print(f"  K_c (used by partial):  {K_c.shape}")
    print(f"  V_c (used by partial):  {V_c.shape}")
    print(f"  LSE_partials (out partial/in reduct): {LSE_partials.shape}, {LSE_partials.dtype}")
    print(f"  O_partials (out partial/in reduct):   {O_partials.shape}, {O_partials.dtype}")
    print(f"  O_final (out reduct):   {O_final.shape}, {O_final.dtype}")


    return Q_in, K_c, V_c, LSE_partials, O_partials, O_final

def generate_partial_instructions_and_timings(num_blocks: int, h_kv: int, num_partials: int, layer_idx: int):
    '''Generates instructions for the GQA Partial kernel.'''
    instructions = [[] for _ in range(num_blocks)]
    instruction_count = 0
    block_idx = 0

    # Instruction format: [opcode, layer_idx, kv_head_idx, num_partials, partial_idx]
    for p in range(num_partials):
        for h in range(h_kv):
            inst = [GQA_PARTIAL_OPCODE, layer_idx, h, num_partials, p] + [0] * (INSTRUCTION_WIDTH - 5)
            instructions[block_idx].append(inst)
            instruction_count += 1
            block_idx = (block_idx + 1) % num_blocks # Distribute instructions

    # Pad instructions
    max_instructions = max(len(block_inst) for block_inst in instructions) if instruction_count > 0 else 0
    total_instructions_padded = 0
    for i in range(num_blocks):
        while len(instructions[i]) < max_instructions:
            instructions[i].append([0] * INSTRUCTION_WIDTH) # Pad with NOPs
        total_instructions_padded += len(instructions[i])


    instructions_tensor = torch.tensor(instructions, dtype=torch.int32, device=TORCH_DEVICE)
    # Ensure timings match the padded instruction shape
    timings_tensor = torch.zeros((num_blocks, max_instructions, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)

    print("\nPartial Kernel Instructions:")
    print(f"  Shape: {instructions_tensor.shape}")
    print(f"  Number of active instructions: {instruction_count}")

    return instructions_tensor, timings_tensor

def generate_reduction_instructions_and_timings(num_blocks: int, h_q: int, num_partials: int):
    '''Generates instructions for the GQA Reduction kernel.'''
    instructions = [[] for _ in range(num_blocks)]
    instruction_count = 0

    # Instruction format: [opcode, num_partials]
    # Assign one reduction instruction per Q head to the first H_q blocks
    for i in range(h_q):
         # Only need one instruction per block for reduction
        inst = [GQA_REDUCTION_OPCODE, num_partials] + [0] * (INSTRUCTION_WIDTH - 2)
        instructions[i].append(inst)
        instruction_count += 1

    # Pad instructions - all blocks need at least one instruction slot
    max_instructions = 1 # We only issue one instruction per active block
    total_instructions_padded = 0
    for i in range(num_blocks):
        while len(instructions[i]) < max_instructions:
            instructions[i].append([0] * INSTRUCTION_WIDTH) # Pad with NOPs
        total_instructions_padded += len(instructions[i])


    instructions_tensor = torch.tensor(instructions, dtype=torch.int32, device=TORCH_DEVICE)
    # Ensure timings match the padded instruction shape
    timings_tensor = torch.zeros((num_blocks, max_instructions, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)

    print("\nReduction Kernel Instructions:")
    print(f"  Shape: {instructions_tensor.shape}")
    print(f"  Number of active instructions: {instruction_count}")

    return instructions_tensor, timings_tensor


# --- Main Script ---

# 1. Generate Inputs
print('\n--- Generating Inputs ---')
Q_in, K_c, V_c, LSE_partials, O_partials, O_final = generate_tensor_inputs(
    L, MAX_PARTIALS, N_max, H_q, H_kv, D_h
)
partial_instructions, partial_timings = generate_partial_instructions_and_timings(
    NUM_BLOCKS, H_kv, NUM_PARTIALS, LAYER_IDX
)

# 2. Run gqa_partial Kernel
print('\n--- Running gqa_partial Kernel ---')
gqa_partial(
    partial_instructions, partial_timings,
    Q_in, K_c, V_c,
    LSE_partials, # Output partial LSE (shape 1, H_q, M_a)
    O_partials,   # Output partial O   (shape 1, H_q, M_a, D_h)
    POS_ID, ATTN_SCALE
)

torch.cuda.synchronize(TORCH_DEVICE)
print("gqa_partial kernel finished.")
print("LSE Shape:", LSE_partials.shape)
for elem in LSE_partials:
    print(elem)
print("O Shape:", O_partials.shape)
for elem in O_partials:
    print(elem)
# Note: LSE_partials and O_partials are now populated and have the correct
# shape and type (float32) expected by the reduction kernel's input layout.

# 3. Generate Reduction Instructions
print('\n--- Generating Reduction Instructions ---')
reduction_instructions, reduction_timings = generate_reduction_instructions_and_timings(
    H_q, H_q, NUM_PARTIALS
)

# 4. Run gqa_reduction Kernel
print('\n--- Running gqa_reduction Kernel ---')
gqa_reduction(
    reduction_instructions, reduction_timings,
    LSE_partials, # Input partial LSE (shape 1, H_q, M_a)
    O_partials,   # Input partial O   (shape 1, H_q, M_a, D_h)
    O_final       # Output final O    (shape 1, 1, H_q, D_h, bf16)
)
torch.cuda.synchronize(TORCH_DEVICE)
print("gqa_reduction kernel finished.")
print("O Final Shape:", O_final.shape)
for elem in O_final:
    print(elem)

# 5. Run Reference Implementation (Full Attention)
print('\n--- Running Reference Full Attention ---')
# This part calculates the ground truth O_ref using standard PyTorch attention
seq_len = POS_ID + 1
K_ref = K_c[LAYER_IDX, :seq_len, :, :] # (seq_len, H_kv, D_h)
V_ref = V_c[LAYER_IDX, :seq_len, :, :] # (seq_len, H_kv, D_h)
O_ref = torch.zeros_like(Q_in, dtype=torch.float32) # Reference output in float32

for h in range(H_kv):
    q_start = h * (H_q // H_kv) # Calculate Q head start index for this group
    q_end = (h + 1) * (H_q // H_kv) # Calculate Q head end index for this group
    
    Q_group = Q_in[q_start:q_end, :]   # (num_q_per_group, D_h)
    K_h = K_ref[:, h, :]               # (seq_len, D_h)
    V_h = V_ref[:, h, :]               # (seq_len, D_h)

    # Calculate attention scores
    # (num_q_per_group, D_h) @ (D_h, seq_len) -> (num_q_per_group, seq_len)
    QK = torch.matmul(Q_group.float(), K_h.float().transpose(-1, -2))
    scaled_QK = QK * ATTN_SCALE

    # Apply softmax
    softmax_QK = torch.softmax(scaled_QK, dim=-1)

    # Calculate output
    # (num_q_per_group, seq_len) @ (seq_len, D_h) -> (num_q_per_group, D_h)
    O_ref[q_start:q_end, :] = torch.matmul(softmax_QK.to(V_h.dtype), V_h)

print("Reference calculation finished.")
print(f"  Reference O_ref shape: {O_ref.shape}, dtype: {O_ref.dtype}")


# 6. Compare Kernel Output with Reference
print('\n--- Comparing Reduction Kernel Output with Reference ---')

# Extract the meaningful part of the kernel's output O_final
# O_final shape is (1, 1, H_q, D_h), bf16. Extract and convert to float32.
O_final_kernel = O_final[0, :, 0, :].float()

print(f"  Kernel O_final shape (extracted): {O_final_kernel.shape}, dtype: {O_final_kernel.dtype}")

# Calculate differences
abs_diff = torch.abs(O_final_kernel - O_ref)
max_abs_err = torch.max(abs_diff)
mean_abs_err = torch.mean(abs_diff)

print(f"\nMax Absolute Error: {max_abs_err.item()}")
print(f"Mean Absolute Error: {mean_abs_err.item()}")

# Add a tolerance check (adjust tolerance as needed)
tolerance = 1e-2 # Tolerance for bfloat16 comparisons
if max_abs_err < tolerance:
    print("\n✅ Test Passed (within tolerance)")
else:
    print(f"\n❌ Test Failed (Max error {max_abs_err.item()} exceeds tolerance {tolerance})")

# Optional: Print some values for debugging
# print("\nSample Values (Kernel):")
# print(O_final_kernel[0, :5])
# print("\nSample Values (Reference):")
# print(O_ref[0, :5])