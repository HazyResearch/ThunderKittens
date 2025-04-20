import math

from einops import einsum
import torch
from gqa_partial import gqa_partial

TORCH_DEVICE = torch.device('cuda:7')

NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_PARTIAL_OPCODE = 1
ATTN_BLOCK_SIZE = 16

# For testing
NUM_PARTIALS = 2
PARTIAL_IDX = 1
LAYER_IDX = 7
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

    Q   = torch.randn(H_q, D_h,            dtype=torch.bfloat16, device=TORCH_DEVICE)
    K_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    L   = torch.zeros(M_a, H_q,            dtype=torch.float32,  device=TORCH_DEVICE)
    O   = torch.zeros(M_a, H_q, D_h,       dtype=torch.bfloat16, device=TORCH_DEVICE)

    return Q, K_c, V_c, L, O

def generate_instructions_and_timings():

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

    return instructions, timings

# Llama 3.2 1B Model Parameters
L = 16 # number of hidden layers
B = 1 # this is fixed for now
N_max = 131072 # maximum sequence length
H_q = 32 # number of query heads
H_kv = 8 # number of key/value heads (number of query groups)
D_h = 64 # dimension of each head

# Kernel parameters
pos_id = 1023
max_pattn_blk = 1024
attn_scale = 1 / math.sqrt(D_h)

# Generate inputs
print('\nGenerating inputs...')
Q, K_c, V_c, L, O = generate_tensor_inputs(L, max_pattn_blk, N_max, H_q, H_kv, D_h)
instructions, timings = generate_instructions_and_timings()

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Timings shape:', timings.shape) 
print('Q shape:', Q.shape)
print('K_c shape:', K_c.shape)
print('V_c shape:', V_c.shape)
print('L shape:', L.shape)
print('O shape:', O.shape)
print('\nRunning the kernel...')
gqa_partial(
    instructions, timings, 
    Q, K_c, V_c, L, O, 
    pos_id, attn_scale
)
torch.cuda.synchronize(TORCH_DEVICE)

# Run the reference implementation
print('\nRunning the reference implementation...')
kv_block_size = ATTN_BLOCK_SIZE
seq_len = pos_id + 1

total_blocks = math.ceil(seq_len / kv_block_size)
blocks_per_partial = math.ceil(total_blocks / NUM_PARTIALS)
start_block = PARTIAL_IDX * blocks_per_partial
end_block = min(start_block + blocks_per_partial, total_blocks)
start_token = start_block * kv_block_size
end_token = min(end_block * kv_block_size, seq_len)

H_q_IDX = H_kv_IDX * 4
Qi = Q[H_q_IDX:H_q_IDX+4, :]
Kj = K_c[LAYER_IDX, start_token:end_token, H_kv_IDX, :]
Vj = V_c[LAYER_IDX, start_token:end_token, H_kv_IDX, :]
QiKj = torch.matmul(Qi, Kj.transpose(-1, -2))
scaled_QiKj = QiKj * attn_scale
softmax = torch.softmax(scaled_QiKj, dim=-1)
# L_ref = torch.logsumexp(scaled_QiKj, dim=-1)
L_ref = torch.log2(torch.sum(torch.exp(scaled_QiKj.float()), dim=-1)) # use log2 consistently
O_ref = torch.matmul(softmax, Vj)

# Verify the output
print('\nComparing outputs...')
print('Qi shape:', Qi.shape)
print('Kj shape:', Kj.shape)
print('Vj shape:', Vj.shape)
print('QiKj shape:', QiKj.shape)
print('scaled_QiKj shape:', scaled_QiKj.shape)
print('softmax shape:', softmax.shape)
print('L_ref shape:', L_ref.shape)
print('O_ref shape:', O_ref.shape)
print(torch.max(torch.abs(O[0, H_q_IDX:H_q_IDX+4, :] - O_ref)))
print(torch.mean(torch.abs(O[0, H_q_IDX:H_q_IDX+4, :] - O_ref)))
print(torch.max(torch.abs(L[0, H_q_IDX:H_q_IDX+4] - L_ref)))
print(torch.mean(torch.abs(L[0, H_q_IDX:H_q_IDX+4] - L_ref)))
