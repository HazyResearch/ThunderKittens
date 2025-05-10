import math

import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cpu')  # Temporarily use CPU instead of CUDA
print(f'Using device: {TORCH_DEVICE}')

# Fixed paramters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_ATTENTION_OPCODE = 2
NUM_OPS = 6
KV_PAGE_SIZE = 128  # Size of each KV cache page

# Llama 70B Model Parameters
B = 1  # this is fixed for now
N_max = 131072  # maximum sequence length
H_q = 64  # number of query heads
H_kv = 8  # number of key/value heads (number of query groups)
D_h = 128  # dimension of each head

# Kernel parameters
LAYER_IDX = 3
POS_ID = 1050
ATTN_SCALE = 1 / math.sqrt(D_h)
ATTN_BLOCK_SIZE = 16
BATCH_IDX = 6

# Unit testing
H_kv_IDX = 5

def generate_tensor_inputs(N_max: int, H_q: int, H_kv: int, D_h: int):
    '''Generate tensor inputs for the attention kernel.
    Args:
        N_max (int): Maximum sequence length.
        H_q (int): Number of query heads.
        H_kv (int): Number of key/value heads (number of query groups).
        D_h (int): Dimension of each head.
    '''
    torch.manual_seed(42)

    # Calculate number of pages needed
    num_pages = (N_max + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE

    Q   = torch.randn(H_q * D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    # KV cache in paged format: (max_num_pages, page_size, num_heads, head_dim)
    K_c = torch.randn(num_pages, KV_PAGE_SIZE, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(num_pages, KV_PAGE_SIZE, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    O   = torch.zeros(H_q, D_h, dtype=torch.float32, device=TORCH_DEVICE)

    return Q, K_c, V_c, O

def generate_itb(): # instruction, timings, barriers
    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Calculate which pages we need to access based on POS_ID
    num_pages_needed = (POS_ID + 1 + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE
    kv_indices = list(range(num_pages_needed))  # List of page indices to access

    # Single instruction, for testing (L, H_kv index, num_kv_indices, kv_indices...)
    instruction = [GQA_ATTENTION_OPCODE, LAYER_IDX, H_kv_IDX, len(kv_indices)] + kv_indices
    instruction += [0] * (INSTRUCTION_WIDTH - len(instruction))  # Pad to INSTRUCTION_WIDTH
    instructions[0].append(instruction)
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
    barriers = torch.zeros((7, NUM_OPS, H_q + 2 * H_kv), dtype=torch.uint32).to(device=TORCH_DEVICE)

    # Fill in the barrier
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 0] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 1] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 2] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 3] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 4] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 5] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 6] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_kv_IDX * 8 + 7] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_q + H_kv_IDX] = 4
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_q + H_kv + H_kv_IDX] = 4

    return instructions, timings, barriers

# Generate inputs
print('\nGenerating inputs...')
instructions, timings, barriers = generate_itb()
Q, K_c, V_c, O = generate_tensor_inputs(N_max, H_q, H_kv, D_h)

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Barrier shape:', barriers.shape)
print('Timings shape:', timings.shape) 
print('Q shape:', Q.shape)
print('K_c shape:', K_c.shape)
print('V_c shape:', V_c.shape)
print('O shape:', O.shape)
print('\nRunning the kernel...')
# kvm_llama(
#     instructions, 
#     barriers, 
#     timings,
#     K_c, 
#     V_c, 
#     Q,
#     O, 
#     POS_ID, 
#     ATTN_SCALE
# )
# torch.cuda.synchronize(TORCH_DEVICE)

# Reshape to match the reference implementation
Q = Q.view(H_q, D_h)    # (H_q, D_h)
O = O.view(H_q, D_h)    # (H_q, D_h)

# Run the reference implementation
print('\nRunning the reference implementation...')
seq_len = POS_ID + 1

H_q_IDX = H_kv_IDX * 8
Qi = Q[H_q_IDX:H_q_IDX+8, :]

# Get the relevant pages and concatenate them
num_pages_needed = (seq_len + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE
K_pages = []
V_pages = []
for page_idx in range(num_pages_needed):
    K_pages.append(K_c[page_idx, :, H_kv_IDX, :])
    V_pages.append(V_c[page_idx, :, H_kv_IDX, :])

print('K_pages shape:', K_pages[0].shape)
print('V_pages shape:', V_pages[0].shape)

# Concatenate pages and trim to seq_len
Kj = torch.cat(K_pages, dim=0)[:seq_len]
Vj = torch.cat(V_pages, dim=0)[:seq_len]

QiKj = torch.matmul(Qi.float(), Kj.float().transpose(-1, -2))
scaled_QiKj = QiKj * ATTN_SCALE
softmax = torch.softmax(scaled_QiKj, dim=-1)
O_ref = torch.matmul(softmax.float(), Vj.float())

# Verify the output
print('\nComparing outputs...')
print('Qi shape:', Qi.shape)
print('Kj shape:', Kj.shape)
print('Vj shape:', Vj.shape)
print('QiKj shape:', QiKj.shape)
print('scaled_QiKj shape:', scaled_QiKj.shape)
print('softmax shape:', softmax.shape)
print('O_ref shape:', O_ref.shape)
print(torch.max(torch.abs(O[H_q_IDX:H_q_IDX+8, :] - O_ref)))
print(torch.mean(torch.abs(O[H_q_IDX:H_q_IDX+8, :] - O_ref)))
print(barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1])
