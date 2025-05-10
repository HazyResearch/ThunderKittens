import math
import torch
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {TORCH_DEVICE}")

# Fixed parameters
NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
QKV_ROPE_APPEND_OPCODE = 2
NUM_OPS = 7

# Llama 70B Model Parameters
HIDDEN_DIM = 8192
HEAD_DIM = 128
NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 8
BATCH_SIZE = 128
KV_BLOCK_SIZE = HEAD_DIM  # Changed to match kernel's behavior of processing entire heads
MAX_NUM_PAGES = 256  # Added to match FlashInfer format
PAGE_SIZE = 128  # Each page stores one sequence position

# Kernel parameters
LAYER_IDX = 0
QKV_BLOCK_IDX = 63  # This now represents which head we're processing
POS_ID = 0
LN_EPS = 1e-5

def generate_tensor_inputs():
    """
    Generates input tensors for the qkv_rope_append kernel.
    """
    torch.manual_seed(42)
    # rms_rope_intermediates = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE) * 0.1
    # qkv_weights = torch.randn((NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE) * 0.1

    # Create rms_rope_intermediates with 1s in first row, 0s elsewhere
    rms_rope_intermediates = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    rms_rope_intermediates[:, 0] = 1.0  # First row is all 1s
    
    # INIT METHOD 2
    # # Create qkv_weights with head indices in ALL rows of each block
    # qkv_weights = torch.zeros((NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    
    # # # For Q heads
    # for i in range(NUM_ATTENTION_HEADS):
    #     qkv_weights[i * HEAD_DIM:(i + 1) * HEAD_DIM, 0] = i  # All rows of each Q head block is the head index
    
    # # For K heads
    # for i in range(NUM_KV_HEADS):
    #     idx = NUM_ATTENTION_HEADS + i
    #     qkv_weights[idx * HEAD_DIM:(idx + 1) * HEAD_DIM, 0] = idx  # All rows of each K head block is the head index
    
    # # For V heads
    # for i in range(NUM_KV_HEADS):
    #     idx = NUM_ATTENTION_HEADS + NUM_KV_HEADS + i
    #     qkv_weights[idx * HEAD_DIM:(idx + 1) * HEAD_DIM, 0] = idx  # All rows of each V head block is the head index

    
    # INIT METHOD 3
    # qkv_weights = torch.zeros((NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN_DIM, 
    #                       dtype=torch.bfloat16, device=TORCH_DEVICE)

    # # Set the first column to be the row indices
    # row_indices = torch.arange(qkv_weights.shape[0], device=qkv_weights.device)
    # print('row_indices:', row_indices)
    # raise Exception('Stop here')
    # qkv_weights[:, 0] = (row_indices * 0.1).to(torch.bfloat16)
    # print('qkv_weights:', qkv_weights)

    # Init method 4
    qkv_weights = torch.zeros((NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN_DIM, 
                          dtype=torch.bfloat16, device=TORCH_DEVICE)

    # Create a pattern that repeats 0-127 for each block of 128 rows
    total_rows = qkv_weights.shape[0]
    block_size = 128

    # Do the modulo operation in int32 first, then convert to bfloat16
    local_indices = torch.arange(total_rows, dtype=torch.int32, device=qkv_weights.device) % block_size
    local_indices = local_indices.to(dtype=qkv_weights.dtype)

    # Set the first column to these local indices
    qkv_weights[:, 0] = local_indices

    # Verify the last few values
    print(local_indices[-10:])
    

    
    # Initialize KV caches in FlashInfer format
    k_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    v_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    
    # Generate rope_cos and rope_sin with appropriate scaling
    rope_cos = torch.cos(torch.linspace(0, 2 * math.pi, HEAD_DIM // 2, device=TORCH_DEVICE)).unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    rope_sin = torch.sin(torch.linspace(0, 2 * math.pi, HEAD_DIM // 2, device=TORCH_DEVICE)).unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    
    q_post_rope = torch.zeros(BATCH_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=TORCH_DEVICE)
    
    # Generate routing table - each batch index maps to a unique page index
    routing_table = torch.zeros(BATCH_SIZE * 2, dtype=torch.int32, device=TORCH_DEVICE)
    routing_table[::2] = torch.arange(BATCH_SIZE-1, -1, -1, dtype=torch.int32, device=TORCH_DEVICE)
    
    return rms_rope_intermediates, qkv_weights, k_cache, v_cache, rope_cos, rope_sin, q_post_rope, routing_table

def generate_itb():
    """
    Generates instructions, timings, and barriers.
    """
    instructions = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    timings = torch.zeros((NUM_BLOCKS, 1, TIMING_WIDTH), dtype=torch.int32, device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, NUM_ATTENTION_HEADS + 2 * NUM_KV_HEADS), dtype=torch.uint32, device=TORCH_DEVICE)

    # Fill instruction: [opcode, layer, qkv_block_idx, pos_id, arg3, arg4, <padding>]
    instruction_values = [
        QKV_ROPE_APPEND_OPCODE,  # opcode
        LAYER_IDX,               # arg1: layer
        0,                       # arg2: row
        QKV_BLOCK_IDX,           # arg3: col
        HIDDEN_DIM // 128,       # arg4: iters
    ]
    for i, val in enumerate(instruction_values):
        instructions[0, 0, i] = val

    return instructions, timings, barriers

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

def main():
    print('\nGenerating inputs...')
    instructions, timings, barriers = generate_itb()
    rms_rope_intermediates, qkv_weights, k_cache, v_cache, rope_cos, rope_sin, q_post_rope, routing_table = generate_tensor_inputs()

    # Run the kernel
    print('Barrier shape:', barriers.shape)
    print('Instruction shape:', instructions.shape)
    print('Timings shape:', timings.shape)
    print('qkv_weights shape:', qkv_weights.shape)
    print('k_cache shape:', k_cache.shape)
    print('v_cache shape:', v_cache.shape)
    print('rope_cos shape:', rope_cos.shape)
    print('rope_sin shape:', rope_sin.shape)
    print('rms_rope_intermediates shape:', rms_rope_intermediates.shape)
    print('q_post_rope shape:', q_post_rope.shape)
    print('routing_table shape:', routing_table.shape)
    print('\nRunning the kernel...')

    kvm_llama(
        barriers,
        instructions,
        timings,
        qkv_weights,
        k_cache,
        v_cache,
        rope_cos,
        rope_sin,
        rms_rope_intermediates,
        q_post_rope,
        routing_table,
        POS_ID
    )
    torch.cuda.synchronize(TORCH_DEVICE)

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

if __name__ == "__main__":
    main()

