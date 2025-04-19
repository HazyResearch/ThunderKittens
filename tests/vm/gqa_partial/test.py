import math

import torch
from gqa_partial import gqa_partial

NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_PARTIAL_OPCODE = 1

# For testing
LAYER_IDX = 7
H_kv_IDX = 0

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

    Q   = torch.randn(H_q, D_h,            dtype=torch.bfloat16, device='cuda:0')
    K_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device='cuda:0')
    V_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device='cuda:0')
    L   = torch.zeros(M_a, H_q,            dtype=torch.bfloat16, device='cuda:0')
    O   = torch.zeros(M_a, H_q, D_h,       dtype=torch.bfloat16, device='cuda:0')

    return Q, K_c, V_c, L, O

def generate_instructions_and_timings():

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Single instruction, for testing (L, H_kv index, num_partials, partial_idx)
    instructions[0].append([GQA_PARTIAL_OPCODE, LAYER_IDX, H_kv_IDX, 2, 0] + [0] * (INSTRUCTION_WIDTH - 5))
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
    instructions = torch.tensor(instructions, dtype=torch.int32).to(device=0)
    timings = torch.zeros((NUM_BLOCKS, instruction_idx // NUM_BLOCKS, TIMING_WIDTH), dtype=torch.int32).to(device=0)

    return instructions, timings

# Llama 3.2 1B Model Parameters
L = 16 # number of hidden layers
B = 1 # this is fixed for now
N_max = 131072 # maximum sequence length
H_q = 32 # number of query heads
H_kv = 8 # number of key/value heads (number of query groups)
D_h = 64 # dimension of each head

# Kernel parameters
pos_id = 1024
attn_blk_size = 16
max_pattn_blk = 1024
softmax_temp = 1 / math.sqrt(D_h)

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
    pos_id, attn_blk_size, softmax_temp
)
torch.cuda.synchronize(0)

# Verify the output
print('\nKernel finished.')
H_q_IDX = H_kv_IDX * 4
O_ref = torch.matmul(torch.matmul(Q[H_q_IDX:H_q_IDX+4, :], K_c[LAYER_IDX, :16, H_kv_IDX, :].T), V_c[LAYER_IDX, :16, H_kv_IDX, :])
print(torch.max(torch.abs(O[0, :4, :] - O_ref)))
print(O[0, :5, :])
