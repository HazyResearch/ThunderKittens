import mla_decode
import torch
import math
from torch.nn.functional import scaled_dot_product_attention as sdpa

torch.manual_seed(0)

# Define opcodes.
OP_STOP, OP_PARTIAL, OP_REDUCTION = 0, 1, 2

# Dimensions and constants.
D_QK, D_VO = 576, 512
PAGE_SIZE = 256
B, NEW_TOKENS, H = 1, 4, 16
LENGTH = 5  # cached sequence length

# --- Instruction Tensor Setup ---

# Partial op layout (16 ints):
#   [opcode, uid, dst_batch, dst_seq, q_batch_idx, q_seq_idx, length, ...padding]
partial_inst = torch.tensor(
    [OP_PARTIAL, 0, -1, 0, 0, 0, LENGTH] + [0] * 9,
    dtype=torch.int32
)

# create one reduction op per new token
# Reduction op layout (16 ints):
#   [opcode, uid, num_iters, dst_batch, dst_seq, src_uid, ...padding]
reduction_insts = []
for token in range(NEW_TOKENS):
    inst = torch.tensor(
        [OP_REDUCTION, 0, 1, 0, token, 0] + [0] * 10,
        dtype=torch.int32
    )
    reduction_insts.append(inst)

# stop op (terminates execution)
stop_inst = torch.tensor([OP_STOP] + [0] * 15, dtype=torch.int32)

insts = [partial_inst] + reduction_insts + [stop_inst]
Ops = torch.stack(insts, dim=0)
Stops = torch.zeros_like(Ops[0])
pad = Stops.unsqueeze(0).repeat(130, 1)
Ops = torch.cat([Ops, pad], dim=0).cuda().contiguous()


softmax_scale = 1.0 / math.sqrt(D_QK)
NUM_PAGES = 100
T = 1     
cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()

# Query tensor: shape (B, NEW_TOKENS, H, D_QK)
query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()

# seq length
lengths = torch.full((B,), LENGTH, dtype=torch.int32, device='cuda')
sizes = (lengths + (PAGE_SIZE - 1)) // PAGE_SIZE
total = lengths.sum().item()
maximum = lengths.max().item()

# block table of shape (B, T).
table = torch.zeros(B, T, dtype=torch.int32).cuda()
sequence_ids, position_ids = (
    torch.arange(T, dtype=torch.int32, device='cuda')[None, :]
    .expand(B, -1)
    .lt(sizes.view(-1, 1))
    .nonzero(as_tuple=True)
)
perm = torch.randperm(NUM_PAGES, device='cuda')[:sequence_ids.size(0)]
perm_sorted = perm.sort().values.int()
table[sequence_ids, position_ids] = perm_sorted

# prefill latent cache and build padded key/value
latent = torch.randn(total, 1, D_QK, dtype=torch.bfloat16).cuda() * 10
expanded = latent.expand(total, H, D_QK)
padded_key = torch.zeros(B, maximum, H, D_QK, dtype=torch.bfloat16).cuda()
padded_value = torch.zeros(B, maximum, H, D_VO, dtype=torch.bfloat16).cuda()
sequence_ids, position_ids = (
    torch.arange(maximum, device='cuda')[None, :]
    .expand(B, -1)
    .lt(lengths.view(-1, 1))
    .nonzero(as_tuple=True)
)
padded_key[sequence_ids, position_ids] = expanded
padded_value[sequence_ids, position_ids] = expanded[..., :D_VO]
entry_ids  = table[sequence_ids, (position_ids // PAGE_SIZE)]
column_ids = position_ids % PAGE_SIZE
cache[entry_ids, column_ids] = latent

# attention mask
if True:
    bounds = (
        torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :]
        + lengths[:, None]
        - NEW_TOKENS
    )
    mask = (
        torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
        .expand(B, H, NEW_TOKENS, -1)
        .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1))
    )
else:
    mask = None


O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()

O_scratch = torch.zeros((B, NEW_TOKENS, 16, D_VO), dtype=torch.float32).cuda()
Lvec_scratch = torch.zeros((B, NEW_TOKENS, 16), dtype=torch.float32).cuda()

Semaphore = torch.zeros((B, NEW_TOKENS), dtype=torch.int32).cuda()


print("Starting kernel execution")
mla_decode.mla_decode(
    Ops,
    query,
    cache.view(B, NUM_PAGES, PAGE_SIZE, D_QK),
    table,
    O,
    O_scratch,
    Lvec_scratch,
    Semaphore,
    softmax_scale
)
torch.cuda.synchronize()
print("Kernel execution finished.")


ref = sdpa(
    query=query.transpose(1, 2),
    key=padded_key.transpose(1, 2),
    value=padded_value.transpose(1, 2),
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
    scale=softmax_scale,
    enable_gqa=False,
).transpose(1, 2)

print("Reference result:", ref)
print("Kernel result:", O)
print("Reference mean:", torch.mean(ref.abs()))
print("Kernel result mean:", torch.mean(O.abs()))
print("Max absolute difference:", torch.max(torch.abs(O - ref)))
print("Average absolute difference:", torch.mean(torch.abs(O - ref)))
