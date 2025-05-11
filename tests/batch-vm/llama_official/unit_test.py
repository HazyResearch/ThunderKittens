import torch
from kvm_llama import kvm_llama


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
HIDDEN_DIM = 8192
INTERMEDIATE_DIM = 28672
HEAD_DIM = 128
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 131072
NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 8
KV_BLOCK_SIZE = HEAD_DIM
MAX_NUM_PAGES = 256
PAGE_SIZE = 128


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
k_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
v_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
rope_cos = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=device)
rope_sin = torch.randn(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32, device=device)
hidden_states = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
rms_rope_intermediates = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
rms_gate_intermediates = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
gate_silu_intermediates = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device)
q_post_rope = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
attn_out = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
silu_out = torch.zeros(BATCH_SIZE, INTERMEDIATE_DIM, dtype=torch.bfloat16, device=device)
logits = torch.zeros(BATCH_SIZE, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
routing_table = torch.zeros(BATCH_SIZE * 2, dtype=torch.int32, device=device)
pos_id = 10
attn_scale = 1 / (HEAD_DIM ** 0.5)
rms_norm_eps = 1e-5

# Generate instructions
instructions = [[] for _ in range(SM_COUNT)]
instructions[0] = [[
    QKV_ROPE_APPEND_OPCODE,  # opcode
    0,                       # arg1: layer
    0,                       # arg2: row
    0,                       # arg3: col
    HIDDEN_DIM // 128,       # arg4: iters
] + [0] * (INSTRUCTION_WIDTH - 5)]

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

# Fill in routing table
routing_table[::2] = torch.arange(BATCH_SIZE-1, -1, -1, dtype=torch.int32, device=device)


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
    routing_table,
    pos_id,
    attn_scale,
    rms_norm_eps
)
torch.cuda.synchronize()

quit()




def run_reference_implementation(rms_rope_intermediates, qkv_weights, k_cache_ref, v_cache_ref, rope_cos, rope_sin, q_post_rope_ref, routing_table):
    """Runs the reference implementation for comparison."""
    # Project to QKV
    qkv = torch.matmul(rms_rope_intermediates, qkv_weights.T)
    
    # Split into Q, K, V
    q = qkv[:, :NUM_ATTENTION_HEADS * HEAD_DIM]
    k = qkv[:, NUM_ATTENTION_HEADS * HEAD_DIM:NUM_ATTENTION_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM]
    v = qkv[:, NUM_ATTENTION_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM:]
    
    # Reshape K and V to match FlashInfer format
    k = k.view(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM)  # [BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM]
    v = v.view(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM)  # [BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM]
    
    # Store K and V in their respective pages
    page_indices = routing_table[::2][:BATCH_SIZE]  # Get every other element up to BATCH_SIZE
    for batch_idx in range(BATCH_SIZE):
        page_idx = page_indices[batch_idx]
        slot_idx = 0
        k_cache_ref[page_idx, slot_idx, :, :] = k[batch_idx]  # Explicitly show all dimension
        v_cache_ref[page_idx, slot_idx, :, :] = v[batch_idx]  # Explicitly show all dimensions

    # Just copy first 8192 elements of qkv
    q_post_rope_ref.copy_(q)



torch.cuda.synchronize()

# Run the reference implementation
print('\nRunning the reference implementation...')
q_post_rope_ref = torch.zeros_like(q_post_rope)
k_cache_ref = torch.zeros_like(k_cache)
v_cache_ref = torch.zeros_like(v_cache)

run_reference_implementation(
    rms_rope_intermediates,
    qkv_weights,
    k_cache_ref,
    v_cache_ref,
    rope_cos,
    rope_sin,
    q_post_rope_ref,
    routing_table
)

# Verify the outputs
print('\nComparing outputs...')
head_idx = QKV_BLOCK_IDX
start_idx = head_idx * HEAD_DIM
end_idx = start_idx + HEAD_DIM

page_indices = routing_table[::2][:BATCH_SIZE]  # Get every other element up to BATCH_SIZE

if head_idx < NUM_ATTENTION_HEADS:  # Q
    print(f'Should have been stored to Q head {head_idx}')
    out_kernel = q_post_rope[:, start_idx:end_idx]
    out_ref = q_post_rope_ref[:, start_idx:end_idx]
elif head_idx < NUM_ATTENTION_HEADS + NUM_KV_HEADS:  # K
    head_idx -= NUM_ATTENTION_HEADS
    print(f'Should have been stored to K head {head_idx}')
    # Get the page indices for this batch from the routing table - only even indices contain page numbers
    # For each batch, check the corresponding page in the cache
        
    out_kernel = torch.stack([k_cache[page_idx, 0, head_idx, :] for page_idx in page_indices])
    out_ref = torch.stack([k_cache_ref[page_idx, 0, head_idx, :] for page_idx in page_indices])
else:  # V
    head_idx -= (NUM_ATTENTION_HEADS + NUM_KV_HEADS)
    print(f'Should have been stored to V head {head_idx}')
    # Get the page indices for this batch from the routing table
    # For each batch, check the corresponding page in the cache
    out_kernel = torch.stack([v_cache[page_idx, 0, head_idx] for page_idx in page_indices])
    out_ref = torch.stack([v_cache_ref[page_idx, 0, head_idx] for page_idx in page_indices])

# print('out_kernel:', out_kernel[1][:64])
# print('out_kernel:', out_kernel[1][64:])

print('out_kernel:', out_kernel)
print('out_ref:', out_ref)
print('out_kernel shape:', out_kernel.shape)
print('out_ref shape:', out_ref.shape)
print('Max absolute difference:', torch.max(torch.abs(out_kernel - out_ref)))
print('Mean absolute difference:', torch.mean(torch.abs(out_kernel - out_ref)))
print('Barrier value:', barriers[LAYER_IDX, QKV_ROPE_APPEND_OPCODE - 1, 0])
