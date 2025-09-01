import torch
from reference import layer_norm_matvec_rope_append
from rms_matvec_rope_append import rms_matvec_rope_append

TORCH_DEVICE = torch.device('cuda:2')

# Fixed paramters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
RMS_MATVEC_ROPE_APPEND_OPCODE = 1
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
LN_EPS = 1e-5

# Unit testing
QKV_BLOCK_IDX = 0
QKV_BLOCK_SIZE = 16

def generate_tensor_inputs():
    torch.manual_seed(42)

    # Inputs
    hidden_states  = torch.randn((H_q * D_h,), dtype=torch.bfloat16, device=TORCH_DEVICE)
    attn_ln_weight = torch.randn((L, H_q * D_h), dtype=torch.bfloat16, device=TORCH_DEVICE)
    qkv_proj       = torch.randn((L, (H_q + 2 * H_kv) * D_h , H_q * D_h), dtype=torch.bfloat16, device=TORCH_DEVICE) # proj weights
    rope_cos       = torch.randn((N_max, D_h), dtype=torch.float32, device=TORCH_DEVICE) # yes I'm being lazy
    rope_sin       = torch.randn((N_max, D_h), dtype=torch.float32, device=TORCH_DEVICE) # yes I'm being lazy

    # Outputs
    post_ln_rope_q = torch.randn((H_q * D_h,), dtype=torch.bfloat16, device=TORCH_DEVICE)
    k_cache        = torch.randn((L, N_max, H_kv, D_h), dtype=torch.bfloat16, device=TORCH_DEVICE)
    v_cache        = torch.randn((L, N_max, H_kv, D_h), dtype=torch.bfloat16, device=TORCH_DEVICE)

    return hidden_states, attn_ln_weight, qkv_proj, rope_cos, rope_sin, post_ln_rope_q, k_cache, v_cache

def generate_itb(): # instruction, timings, barriers

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # Single instruction, for testing (L, H_kv index, num_partials, partial_idx)
    instructions[0].append([RMS_MATVEC_ROPE_APPEND_OPCODE, LAYER_IDX, QKV_BLOCK_IDX] + [0] * (INSTRUCTION_WIDTH - 3))
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
    barriers = torch.zeros((L, NUM_OPS, H_q + 2 * H_kv), dtype=torch.uint32).to(device=TORCH_DEVICE)

    # Set up the barrier
    barriers[LAYER_IDX, RMS_MATVEC_ROPE_APPEND_OPCODE - 1, 0] = 512

    return instructions, timings, barriers

# Generate inputs
print('\nGenerating inputs...')
instructions, timings, barriers = generate_itb()
hidden_states, attn_ln_weight, qkv_proj, rope_cos, rope_sin, post_ln_rope_q, k_cache, v_cache = generate_tensor_inputs()

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Barrier shape:', barriers.shape)
print('Timings shape:', timings.shape)
print('hidden_states shape:', hidden_states.shape)
print('attn_ln_weight shape:', attn_ln_weight.shape)
print('qkv_proj shape:', qkv_proj.shape)
print('rope_cos shape:', rope_cos.shape)
print('rope_sin shape:', rope_sin.shape)
print('post_ln_rope_q shape:', post_ln_rope_q.shape)
print('k_cache shape:', k_cache.shape)
print('v_cache shape:', v_cache.shape)
print('\nRunning the kernel...')
rms_matvec_rope_append(
    instructions,
    barriers,
    timings,
    hidden_states,
    attn_ln_weight,
    qkv_proj,
    rope_cos,
    rope_sin,
    post_ln_rope_q,
    k_cache,
    v_cache,
    POS_ID,
    LN_EPS
)
torch.cuda.synchronize(TORCH_DEVICE)

# Run the reference implementation
print('\nRunning the reference implementation...')
post_ln_rope_q_ref = torch.zeros_like(post_ln_rope_q)
k_cache_ref = torch.zeros_like(k_cache)
v_cache_ref = torch.zeros_like(v_cache)
layer_norm_matvec_rope_append(
    # Globals
    hidden_states=hidden_states,
    attn_ln_weight=attn_ln_weight,
    qkv_proj=qkv_proj,
    rope_cos=rope_cos,
    rope_sin=rope_sin,
    post_ln_rope_q=post_ln_rope_q_ref,
    k_cache=k_cache_ref,
    v_cache=v_cache_ref,
    pos_id=POS_ID,

    # Globals, but fixed
    qkv_block_size=QKV_BLOCK_SIZE,
    num_attention_heads=H_q,
    num_kv_heads=H_kv,
    head_dim=D_h,
    ln_eps=LN_EPS,

    # Instruction
    layer_idx=LAYER_IDX,
    output_block_idx=QKV_BLOCK_IDX,
)

# Verify the outputs
print('\nComparing outputs...')
start_idx = QKV_BLOCK_IDX * QKV_BLOCK_SIZE
end_idx = start_idx + QKV_BLOCK_SIZE
if start_idx < H_q * D_h: # Q
    print(f'Should have been stored to Q[{start_idx}:{end_idx}]')
    out_kernel = post_ln_rope_q[start_idx:end_idx]
    out_ref = post_ln_rope_q_ref[start_idx:end_idx]
elif start_idx < (H_q + H_kv) * D_h: # K
    start_idx -= D_h * H_q
    end_idx -= D_h * H_q
    print(f'Should have been stored to K[{start_idx}:{end_idx}]')
    out_kernel = k_cache[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
    out_ref = k_cache_ref[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
else: # V
    start_idx -= D_h * (H_q + H_kv)
    end_idx -= D_h * (H_q + H_kv)
    print(f'Should have been stored to V[{start_idx}:{end_idx}]')
    out_kernel = v_cache[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
    out_ref = v_cache_ref[LAYER_IDX, POS_ID].view(-1)[start_idx:end_idx]
print('out_kernel shape:', out_kernel.shape)
print('out_ref shape:', out_ref.shape)
print(torch.max(torch.abs(out_kernel - out_ref)))
print(torch.mean(torch.abs(out_kernel - out_ref)))
print(barriers[LAYER_IDX, RMS_MATVEC_ROPE_APPEND_OPCODE])
