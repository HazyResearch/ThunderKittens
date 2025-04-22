import math

import torch

TORCH_DEVICE = torch.device('cuda:5')

# Fixed paramters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
TEMP_OPCODE = 1 # this is not a full instruction

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

# Unit testing
QKV_BLOCK_IDX = 0
QKV_BLOCK_SIZE = 16

def generate_tensor_inputs():
    torch.manual_seed(42)

    QKV_proj = torch.randn((H_q + H_kv * 2) * D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    rope_cos = torch.randn((N_max, D_h), dtype=torch.bfloat16, device=TORCH_DEVICE) # yes I'm being lazy
    rope_sin = torch.randn((N_max, D_h), dtype=torch.bfloat16, device=TORCH_DEVICE) # yes I'm being lazy
    Q   = torch.randn(H_q * D_h,           dtype=torch.bfloat16, device=TORCH_DEVICE)
    K_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(L, N_max, H_kv, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)

    return QKV_proj, rope_cos, rope_sin, Q, K_c, V_c

def generate_instructions_and_timings():

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Single instruction, for testing (L, H_kv index, num_partials, partial_idx)
    instructions[0].append([TEMP_OPCODE, QKV_BLOCK_IDX] + [0] * (INSTRUCTION_WIDTH - 2))
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

# Generate inputs
print('\nGenerating inputs...')
instructions, timings = generate_instructions_and_timings()
QKV_proj, rope_cos, rope_sin, Q, K_c, V_c = generate_tensor_inputs()

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Timings shape:', timings.shape)
print('QKV_proj shape:', QKV_proj.shape)
print('rope_cos shape:', rope_cos.shape)
print('rope_sin shape:', rope_sin.shape)
print('Q shape:', Q.shape)
print('K_c shape:', K_c.shape)
print('V_c shape:', V_c.shape)
print('\nRunning the kernel...')
# RUN KERNEL
torch.cuda.synchronize(TORCH_DEVICE)

# Run the reference implementation
print('\nRunning the reference implementation...')
Q_in = QKV_proj[:H_q * D_h].reshape(H_q, D_h)
K_in = QKV_proj[H_q * D_h:H_q * D_h + H_kv * D_h].reshape(H_kv, D_h)
V_in = QKV_proj[H_q * D_h + H_kv * D_h:].reshape(H_kv, D_h)
rope_cos_at_pos = rope_cos[POS_ID, :].unsqueeze(0)
rope_sin_at_pos = rope_sin[POS_ID, :].unsqueeze(0)
mask = (torch.arange(D_h) % 2 == 0).unsqueeze(0)
Q_rotated = torch.roll(Q_in, 1, dims=-1)
Q_rotated[mask.expand_as(Q_rotated)] *= -1
K_rotated = torch.roll(K_in, 1, dims=-1)
K_rotated[mask.expand_as(K_rotated)] *= -1
Q_roped = Q_in * rope_cos_at_pos + Q_rotated * rope_sin_at_pos
K_roped = K_in * rope_cos_at_pos + K_rotated * rope_sin_at_pos

print('\nComparing outputs...')
start_idx = QKV_BLOCK_IDX * QKV_BLOCK_SIZE
end_idx = start_idx + QKV_BLOCK_SIZE    
if start_idx < H_q * D_h: # Q
    print(f'Should have been stored to Q[{start_idx}:{end_idx}]')
    out_kernel = Q[start_idx:end_idx]
    out_ref = Q_roped.view(-1)[start_idx:end_idx]
elif start_idx < (H_q + H_kv) * D_h: # K
    start_idx -= D_h * H_q
    end_idx -= D_h * H_q
    print(f'Should have been stored to K[{start_idx}:{end_idx}]')
    out_kernel = K_c[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
    out_ref = K_roped.view(-1)[start_idx:end_idx]
else: # V
    start_idx -= D_h * (H_q + H_kv)
    end_idx -= D_h * (H_q + H_kv)
    print(f'Should have been stored to V[{start_idx}:{end_idx}]')
    out_kernel = V_c[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
    out_ref = V_in.view(-1)[start_idx:end_idx]
print('out_kernel shape:', out_kernel.shape)
print('out_ref shape:', out_ref.shape)
print(torch.max(torch.abs(out_kernel - out_ref)))
print(torch.mean(torch.abs(out_kernel - out_ref)))
