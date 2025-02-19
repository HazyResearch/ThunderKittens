import mla_decode
import torch
import math
import torch.nn.functional as F

torch.manual_seed(0)

#----------------------------------------------------------------------------
# Opcodes and dimensions.
OP_STOP      = 0
OP_PARTIAL   = 1
OP_REDUCTION = 2

D_QK = 576
D_VO = 512
PAGE_SIZE = 256
NUM_ROWS = 32

# we simulate a cache containing LENGTH tokens.
LENGTH = 128

# run in non–batch mode so that partial results are written to scratch.
# two partial ops (for two segments) will be issued with unique uids.
uid1 = 100
uid2 = 101

# one batch, one new token, and H=16 heads.
B = 1
NEW_TOKENS = 1
H = 16

# scale factor.
softmax_scale = 1.0 / math.sqrt(D_QK)

#----------------------------------------------------------------------------
# Build the instructions tensor.
#
# The global instructions type is gl<int, 1, -1, -1, 8>
# we create a tensor of shape (1, grid, num_task_iters, 8). 
grid = 132
num_task_iters = 2  # one task iteration for the partial op, one for reduction
instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device='cuda')

# only fill block 0 with work; the rest are stops

# Partial op 1 (for segment 1): placed in block 0, task_iter 0.
instructions[0, 0, 0, :] = torch.tensor(
    [OP_PARTIAL, uid1, -1, uid1, 0, 0, LENGTH, 0],
    dtype=torch.int32, device='cuda')

# Partial op 2 (for segment 2): placed in block 1, task_iter 0.
instructions[0, 1, 0, :] = torch.tensor(
    [OP_PARTIAL, uid2, -1, uid2, 0, 0, LENGTH, 0],
    dtype=torch.int32, device='cuda')

# Reduction op: placed in block 0, task_iter 1, combining results from uid1 and uid2.
instructions[0, 0, 1, :] = torch.tensor(
    [OP_REDUCTION, -1, 0, uid1, uid2, 0, 0, 0],
    dtype=torch.int32, device='cuda')

# The remaining instructions (other blocks/task_iters) remain OP_STOP.

# prepare global buffers
# Q global: expected shape (B, NEW_TOKENS, H, D_QK).
q = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16, device='cuda')

# Cache global: non–batch mode we allocate shape (1, num_pages, PAGE_SIZE, D_QK)
num_pages = 1
cache = torch.randn((1, num_pages, PAGE_SIZE, D_QK), dtype=torch.bfloat16, device='cuda')
# fill the cache with "latent" keys
# create a latent tensor of shape (LENGTH, 1, D_QK)
latent = torch.randn((LENGTH, 1, D_QK), dtype=torch.bfloat16, device='cuda') * 10

# reference we need to build padded_key and padded_value:
#   – expand latent to (LENGTH, H, D_QK)
expanded = latent.expand(LENGTH, H, D_QK)
#   – padded_key: shape (B, LENGTH, H, D_QK)
padded_key = expanded.unsqueeze(0).clone()
#   – padded_value: shape (B, LENGTH, H, D_VO); take the first D_VO channels.
padded_value = expanded[:, :, :D_VO].unsqueeze(0).clone()
# fill cache - for simplicity, assume all tokens go to page 0, positions 0..(LENGTH-1).
cache[0, 0, :LENGTH, :] = latent.squeeze(1)

# table global: for non–batch mode, expected shape (1, 1, max_uid, 1).
max_uid = 1024
table = torch.zeros((1, 1, max_uid, 1), dtype=torch.int32, device='cuda')

# O global: shape (B, NEW_TOKENS, H, D_VO). (Not used in non–batch mode.)
O = torch.empty((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16, device='cuda')

# O_scratch global: declared as gl<float, 1, -1, 64, D_VO, o_tile_fl>
# Expected external shape: (1, max_uid, 64, D_VO)
O_scratch = torch.empty((1, max_uid, 64, D_VO), dtype=torch.float32, device='cuda')

# Lvec_scratch global: expected shape (1, 1, max_uid, 64)
Lvec_scratch = torch.empty((1, 1, max_uid, 64), dtype=torch.float32, device='cuda')

# Semaphore global: expected shape (1, 1, 1, max_uid)
semaphore = torch.zeros((1, 1, 1, max_uid), dtype=torch.int32, device='cuda')

#----------------------------------------------------------------------------
# Print buffer shapes.
print("Instructions shape:", instructions.shape)
print("q shape:", q.shape)
print("cache shape:", cache.shape)
print("table shape:", table.shape)
print("O shape:", O.shape)
print("O_scratch shape:", O_scratch.shape)
print("Lvec_scratch shape:", Lvec_scratch.shape)
print("Semaphore shape:", semaphore.shape)

#----------------------------------------------------------------------------
# Launch the kernel.
print("Launching kernel (partial + reduction run in non–batch mode)...")
mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
torch.cuda.synchronize()
print("Kernel execution finished.")

#----------------------------------------------------------------------------
# In non–batch mode the partial and reduction ops write their results to scratch.
# Our reduction op was set with dst.seq_idx = 0, so we expect its final output
# to appear at scratch coordinate corresponding to uid 0.
#
# The reduction-template stores final results in tiles of type st_fl<16, D_VO> (i.e. 16 rows).
# We assume that a single warp group did the work so that the final output is in O_scratch[0, 0, 0:16, :].
final_out = O_scratch[0, 0, 0:16, :].to(torch.bfloat16)
print("\nFinal output from reduction op (extracted from O_scratch[0,0,0:16,:]):")
print(final_out)

#----------------------------------------------------------------------------
# Compute reference output using PyTorch's scaled_dot_product_attention.
#
# Our query is from q (B, NEW_TOKENS, H, D_QK). For reference, we use the full cached keys.
# Transpose to (B, H, NEW_TOKENS, D_QK) for query and (B, H, LENGTH, D_QK) for keys.
query_t = q.transpose(1, 2)
key_t   = padded_key.transpose(1, 2)
value_t = padded_value.transpose(1, 2)
# Build an attention mask that allows all keys (shape: (B, H, NEW_TOKENS, LENGTH)).
mask = torch.ones((B, H, NEW_TOKENS, LENGTH), device='cuda', dtype=torch.bool)
ref = F.scaled_dot_product_attention(query_t, key_t, value_t,
                                     attn_mask=mask,
                                     dropout_p=0.0,
                                     is_causal=False,
                                     scale=softmax_scale)
# Transpose back to (B, NEW_TOKENS, H, D_VO)
ref = ref.transpose(1, 2)
print("\nReference output (sdpa):")
print(ref)

#----------------------------------------------------------------------------
# Compare final result (from reduction op) with reference.
abs_diff = torch.abs(final_out - ref[0, 0])
print("\nMean absolute difference:", torch.mean(abs_diff).item())
print("Max absolute difference:", torch.max(abs_diff).item())
