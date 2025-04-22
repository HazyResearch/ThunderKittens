import math

import torch
from gqa_partial import gqa_partial

TORCH_DEVICE = torch.device('cuda:2')

# Fixed paramters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_PARTIAL_OPCODE = 1
GQA_REDUCTION_OPCODE = 2
NUM_OPS = 6

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
MAX_PARTIALS = 1024
ATTN_SCALE = 1 / math.sqrt(D_h)
NUM_PARTIALS = 2
ATTN_BLOCK_SIZE = 16

# Unit testing
PARTIAL_IDX = 1
H_kv_IDX = 5

def generate_tensor_inputs(L: int, M_a: int, N_max: int, H_q: int, H_kv: int, D_h: int):
    '''Generate tensor inputs for the GQA kernel.
    Args:
        L (int): Number of hidden layers.
        M_a (int): Maximum partial attention blocks.
        N_max (int): Maximum sequence length.
        H_q (int): Number of query heads.
        H_kv (int): Number of key/value heads (number of query groups).
        D_h (int): Dimension of each head.
    '''
    torch.manual_seed(42)

    Q   = torch.randn(H_q * D_h,           dtype=torch.bfloat16, device=TORCH_DEVICE)
    K_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    LSE = torch.zeros(H_q, M_a,            dtype=torch.float32,  device=TORCH_DEVICE)
    O   = torch.zeros(H_q, M_a, D_h,       dtype=torch.float32,  device=TORCH_DEVICE)

    return Q, K_c, V_c, LSE, O

def generate_itb(): # instruction, timings, barriers

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Single instruction, for testing (L, H_kv index, num_partials, partial_idx)
    instructions[0].append([GQA_PARTIAL_OPCODE, LAYER_IDX, H_kv_IDX, NUM_PARTIALS, PARTIAL_IDX] + [0] * (INSTRUCTION_WIDTH - 5))
    instruction_idx += 1

    # All blocks must have same number of instructions
    max_instructions = -1
    for i in range(NUM_BLOCKS):
        max_instructions = max(max_instructions, len(instructions[i]))
    for i in range(NUM_BLOCKS):
        while len(instructions[i]) < max_instructions:
            instructions[i].append([0] * INSTRUCTION_WIDTH)
        instruction_idx += 1

    # (BlockIdx, InstructionIdx, Instruction)
    # If opcode (instructions[:, :, 0]) is invalid, the instruction is ignored
    instructions = torch.tensor(instructions, dtype=torch.int32).to(device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, instruction_idx // NUM_BLOCKS, TIMING_WIDTH), dtype=torch.int32).to(device=TORCH_DEVICE)
    barriers = torch.zeros((L, NUM_OPS, H_q + 2 * H_kv), dtype=torch.int32).to(device=TORCH_DEVICE)

    # Fill in the barrier
    barriers[LAYER_IDX, GQA_PARTIAL_OPCODE - 1, H_kv_IDX * 4 + 0] = 4
    barriers[LAYER_IDX, GQA_PARTIAL_OPCODE - 1, H_kv_IDX * 4 + 1] = 4
    barriers[LAYER_IDX, GQA_PARTIAL_OPCODE - 1, H_kv_IDX * 4 + 2] = 4
    barriers[LAYER_IDX, GQA_PARTIAL_OPCODE - 1, H_kv_IDX * 4 + 3] = 4
    barriers[LAYER_IDX, GQA_PARTIAL_OPCODE - 1, H_q + H_kv_IDX] = 4

    return instructions, timings, barriers

# Generate inputs
print('\nGenerating inputs...')
instructions, timings, barriers = generate_itb()
Q, K_c, V_c, LSE, O = generate_tensor_inputs(L, MAX_PARTIALS, N_max, H_q, H_kv, D_h)

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Timings shape:', timings.shape) 
print('Q shape:', Q.shape)
print('K_c shape:', K_c.shape)
print('V_c shape:', V_c.shape)
print('LSE shape:', LSE.shape)
print('O shape:', O.shape)
print('\nRunning the kernel...')
gqa_partial(
    instructions, barriers, timings, 
    Q, K_c, V_c, LSE, O, 
    POS_ID, ATTN_SCALE
)
torch.cuda.synchronize(TORCH_DEVICE)

# Reshape to match the reference implementation
Q = Q.view(H_q, D_h)    # (H_q, D_h)
LSE = LSE.permute(1, 0) # (M_a, H_q)
O = O.permute(1, 0, 2)  # (M_a, H_q, D_h)

# Run the reference implementation
print('\nRunning the reference implementation...')
seq_len = POS_ID + 1
total_blocks = math.ceil(seq_len / ATTN_BLOCK_SIZE)
blocks_per_partial = math.ceil(total_blocks / NUM_PARTIALS)
start_block = PARTIAL_IDX * blocks_per_partial
end_block = min(start_block + blocks_per_partial, total_blocks)
start_token = start_block * ATTN_BLOCK_SIZE
end_token = min(end_block * ATTN_BLOCK_SIZE, seq_len)

H_q_IDX = H_kv_IDX * 4
Qi = Q[H_q_IDX:H_q_IDX+4, :]
Kj = K_c[LAYER_IDX, start_token:end_token, H_kv_IDX, :]
Vj = V_c[LAYER_IDX, start_token:end_token, H_kv_IDX, :]
QiKj = torch.matmul(Qi.float(), Kj.float().transpose(-1, -2))
scaled_QiKj = QiKj * ATTN_SCALE
softmax = torch.softmax(scaled_QiKj, dim=-1)
LSE_ref = torch.log2(torch.sum(torch.exp(scaled_QiKj.float()), dim=-1)) # use log2 consistently
O_ref = torch.matmul(softmax.float(), Vj.float())

# Verify the output
print('\nComparing outputs...')
print('Qi shape:', Qi.shape)
print('Kj shape:', Kj.shape)
print('Vj shape:', Vj.shape)
print('QiKj shape:', QiKj.shape)
print('scaled_QiKj shape:', scaled_QiKj.shape)
print('softmax shape:', softmax.shape)
print('LSE_ref shape:', LSE_ref.shape)
print('O_ref shape:', O_ref.shape)
print(torch.max(torch.abs(O[PARTIAL_IDX, H_q_IDX:H_q_IDX+4, :] - O_ref)))
print(torch.mean(torch.abs(O[PARTIAL_IDX, H_q_IDX:H_q_IDX+4, :] - O_ref)))
print(torch.max(torch.abs(LSE[PARTIAL_IDX, H_q_IDX:H_q_IDX+4] - LSE_ref)))
print(torch.mean(torch.abs(LSE[PARTIAL_IDX, H_q_IDX:H_q_IDX+4] - LSE_ref)))
print(barriers[LAYER_IDX, GQA_REDUCTION_OPCODE - 1])
