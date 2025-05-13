import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.set_printoptions(sci_mode=False)

from kvm_llama import kvm_llama
from reference import matvec_rope


###
#  Parameters
###
SM_COUNT = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
BATCH_SIZE = 128
QKV_ROPE_APPEND_OPCODE = 2
NUM_OPS = 8
NUM_LAYERS = 1 # actual value is 80
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 14336
HEAD_DIM = 128
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 4096 # actual value is 131072
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
KV_BLOCK_SIZE = HEAD_DIM
MAX_NUM_PAGES = 256
PAGE_SIZE = 128
QKV_BLOCK_SIZE = 128
LAYER_IDX = 0


###
#   Prepare inputs (follow the order & naming in llama.cuh)
#   Set relevant parameters to randn, and the rest to zeros
###
print('\nGenerating inputs...')
device = torch.device('cuda:0')
torch.manual_seed(42)
Bar = torch.zeros((NUM_LAYERS, NUM_OPS, NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS), dtype=torch.uint32, device=device)
instructions = None # set below
timings = None # set below
qkv_weights = torch.randn(NUM_LAYERS, (NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
attn_norm_weights = torch.zeros(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
o_weights = torch.zeros(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
mlp_norm_weights = torch.zeros(NUM_LAYERS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
up_weights = torch.zeros(NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
gate_weights = torch.zeros(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
down_weights = torch.zeros(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device)
lm_head_norm_weights = torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=device)
lm_head_weights = torch.zeros(VOCAB_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
k_cache = torch.zeros(NUM_LAYERS*BATCH_SIZE, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
v_cache = torch.zeros(NUM_LAYERS*BATCH_SIZE, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=device)
rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=device)
hidden_states = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
rms_rope_intermediates = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
rms_gate_intermediates = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
gate_silu_intermediates = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device)
q_post_rope = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
attn_out = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
silu_out = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device)
logits = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
pos_id = 10
attn_scale = 1 / (HEAD_DIM ** 0.5)
rms_norm_eps = 1e-5

# Generate instructions
instructions = [[] for _ in range(SM_COUNT)]
instruction_idx = 0
for block_idx in range(NUM_ATTENTION_HEADS + NUM_KV_HEADS * 2):
    instructions[instruction_idx] = [[
        QKV_ROPE_APPEND_OPCODE, LAYER_IDX, block_idx
    ] + [0]*(INSTRUCTION_WIDTH - 3)]
    instruction_idx += 1

# Pad instructions
max_instructions = -1
for i in range(SM_COUNT):
    max_instructions = max(max_instructions, len(instructions[i]))
for i in range(SM_COUNT):
    while len(instructions[i]) < max_instructions:
        instructions[i].append([0] * INSTRUCTION_WIDTH)

# Convert to tensor
instructions = torch.tensor(instructions, dtype=torch.int32, device=device)

# Generate timings
timings = torch.zeros((SM_COUNT, instructions.shape[1], TIMING_WIDTH), dtype=torch.int32, device=device)


###
#  Launch the kernel
###
print('\nRunning the kernel...')
torch.cuda.synchronize()
kvm_llama(
    Bar,
    instructions,
    timings,
    qkv_weights,
    attn_norm_weights,
    o_weights,
    mlp_norm_weights,
    up_weights,
    gate_weights,
    down_weights,
    lm_head_norm_weights,
    lm_head_weights,
    k_cache,
    v_cache,
    rope_cos,
    rope_sin,
    hidden_states,
    rms_rope_intermediates,
    rms_gate_intermediates,
    gate_silu_intermediates,
    q_post_rope,
    attn_out,
    silu_out,
    logits,
    pos_id,
    attn_scale,
    rms_norm_eps
)
torch.cuda.synchronize()


###
#  Check correctness
###
q_post_rope_ref, k_post_rope_ref, v_ref = matvec_rope(
    rms_rope_intermediates,
    qkv_weights[LAYER_IDX],
    rope_cos,
    rope_sin,
    pos_id,
    NUM_ATTENTION_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM
)

q_diff = (q_post_rope - q_post_rope_ref).abs()
print(f'Q -- max abs diff: {q_diff.max()}, mean abs diff: {q_diff.mean()}')

k_diff = (k_cache[LAYER_IDX*BATCH_SIZE:(LAYER_IDX+1)*BATCH_SIZE, pos_id] - k_post_rope_ref).abs()
print(f'K -- max abs diff: {k_diff.max()}, mean abs diff: {k_diff.mean()}')

v_diff = (v_cache[LAYER_IDX*BATCH_SIZE:(LAYER_IDX+1)*BATCH_SIZE, pos_id] - v_ref).abs()
print(f'V -- max abs diff: {v_diff.max()}, mean abs diff: {v_diff.mean()}')
