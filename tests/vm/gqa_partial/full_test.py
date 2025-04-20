import math

from einops import einsum
import torch
from gqa_partial import gqa_partial

TORCH_DEVICE = torch.device('cuda:7')

# Fixed paramters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_PARTIAL_OPCODE = 1

# Llama 3.2 1B Model Parameters
L = 16 # number of hidden layers
B = 1 # this is fixed for now
N_max = 131072 # maximum sequence length
H_q = 32 # number of query heads
H_kv = 8 # number of key/value heads (number of query groups)
D_h = 64 # dimension of each head

# Kernel parameters
LAYER_IDX = 3
POS_ID = 1023
MAX_PARTIALS = 1024
ATTN_SCALE = 1 / math.sqrt(D_h)
NUM_PARTIALS = 2
ATTN_BLOCK_SIZE = 16

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
    LSE   = torch.zeros(M_a, H_q,          dtype=torch.float32,  device=TORCH_DEVICE)
    O   = torch.zeros(M_a, H_q, D_h,       dtype=torch.bfloat16, device=TORCH_DEVICE)

    return Q, K_c, V_c, LSE, O

def generate_instructions_and_timings():

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Instruction format: (layer_idx, H_kv_index, num_partials, partial_idx)
    i = 0
    for p in range(NUM_PARTIALS):
        for h in range(H_kv):
            instructions[i].append([GQA_PARTIAL_OPCODE, LAYER_IDX, h, NUM_PARTIALS, p] + [0] * (INSTRUCTION_WIDTH - 5))
            instruction_idx += 1
            i = (i + 1) % NUM_BLOCKS

    # All blocks must have same number of instructions
    max_instructions = -1
    for i in range(NUM_BLOCKS):
        max_instructions = max(max_instructions, len(instructions[i]))
    for i in range(NUM_BLOCKS):
        while len(instructions[i]) < max_instructions:
            instructions[i].append([0] * INSTRUCTION_WIDTH)
        instruction_idx += 1

    # (BlockIdx, InstructionIdx, InstructionOption)
    # If opcode (instructions[:, :, 0]) is invalid, the instruction is ignored
    instructions = torch.tensor(instructions, dtype=torch.int32).to(device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, instruction_idx // NUM_BLOCKS, TIMING_WIDTH), dtype=torch.int32).to(device=TORCH_DEVICE)

    return instructions, timings

# Generate inputs
print('\nGenerating inputs...')
instructions, timings = generate_instructions_and_timings()
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
    instructions, timings, 
    Q, K_c, V_c, LSE, O, 
    POS_ID, ATTN_SCALE
)
torch.cuda.synchronize(TORCH_DEVICE)

# Run the reference implementation
print('\nRunning the reference implementation...')
L_ref = torch.zeros_like(LSE)
O_ref = torch.zeros_like(O)
seq_len = POS_ID + 1
for p in range(NUM_PARTIALS):
    for h in range(H_kv):
        total_blocks = math.ceil(seq_len / ATTN_BLOCK_SIZE)
        blocks_per_partial = math.ceil(total_blocks / NUM_PARTIALS)
        start_block = p * blocks_per_partial
        end_block = min(start_block + blocks_per_partial, total_blocks)
        start_token = start_block * ATTN_BLOCK_SIZE
        end_token = min(end_block * ATTN_BLOCK_SIZE, seq_len)

        H_q_start = h * 4
        Qi = Q[H_q_start:H_q_start+4, :]
        Kj = K_c[LAYER_IDX, start_token:end_token, h, :]
        Vj = V_c[LAYER_IDX, start_token:end_token, h, :]
        QiKj = torch.matmul(Qi, Kj.transpose(-1, -2))
        scaled_QiKj = QiKj * ATTN_SCALE
        softmax = torch.softmax(scaled_QiKj, dim=-1)
        # L_ref[p, H_q_start:H_q_start+4] = torch.logsumexp(scaled_QiKj, dim=-1)
        L_ref[p, H_q_start:H_q_start+4] = torch.log2(torch.sum(torch.exp(scaled_QiKj.float()), dim=-1)) # use log2 consistently
        O_ref[p, H_q_start:H_q_start+4, :] = torch.matmul(softmax, Vj)

        print(f'\nPartial {p}, Head {h}:')
        print(torch.max(torch.abs(O[p, H_q_start:H_q_start+4, :] - O_ref[p, H_q_start:H_q_start+4, :])))
        print(torch.mean(torch.abs(O[p, H_q_start:H_q_start+4, :] - O_ref[p, H_q_start:H_q_start+4, :])))
        print(torch.max(torch.abs(LSE[p, H_q_start:H_q_start+4] - L_ref[p, H_q_start:H_q_start+4])))
        print(torch.mean(torch.abs(LSE[p, H_q_start:H_q_start+4] - L_ref[p, H_q_start:H_q_start+4])))

# # Run the reference implementation
# print('\nRunning the reference implementation...')
# O_ref = torch.zeros((L, H_q, D_h), dtype=torch.bfloat16, device=TORCH_DEVICE)
# seq_len = POS_ID + 1
# for l in range(L):
#     for h in range(H_kv):
#         Q_start = 4 * H_kv
#         Q_end = 4 * (H_kv + 1)
#         K = K_c[l, :seq_len, h, :] # (seq_len, D_h)
#         V = V_c[l, :seq_len, h, :] # (seq_len, D_h)

#         QK = torch.matmul(Q[Q_start:Q_end, :], K.transpose(-2, -1)) # Just use same Q for all layers
#         QK /= (Q.size(-1) ** 0.5)
#         QK = torch.nn.functional.softmax(QK, dim=-1)
#         O_ref[l, Q_start:Q_end, :] = torch.matmul(QK, V) # (H_q, D_h)

# Verify the output
# print('\nComparing outputs...')
# print('Qi shape:', Qi.shape)
# print('Kj shape:', Kj.shape)
# print('Vj shape:', Vj.shape)
# print('QiKj shape:', QiKj.shape)
# print('scaled_QiKj shape:', scaled_QiKj.shape)
# print('softmax shape:', softmax.shape)
# print('L_ref shape:', L_ref.shape)
# print('O_ref shape:', O_ref.shape)
# print(torch.max(torch.abs(O[0, H_q_IDX:H_q_IDX+4, :] - O_ref)))
# print(torch.mean(torch.abs(O[0, H_q_IDX:H_q_IDX+4, :] - O_ref)))
# print(torch.max(torch.abs(LSE[0, H_q_IDX:H_q_IDX+4] - L_ref)))
# print(torch.mean(torch.abs(LSE[0, H_q_IDX:H_q_IDX+4] - L_ref)))
