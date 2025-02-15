import mla_decode
import torch
import math

torch.manual_seed(0)

from torch.nn.functional import linear, scaled_dot_product_attention as sdpa

OP_STOP, OP_PARTIAL, OP_REDUCTION = 0, 1, 2

D_QK, D_VO = 576, 512
PAGE_SIZE = 256

B, NEW_TOKENS, H = 1, 1, 16

# Initialize arguments
# &mla_decode_layout::globals::Ops,
Ops = torch.tensor([
    [
        OP_PARTIAL, # Opcode
        0,          # UID
        0, 0,       # Batch, seq of dst (positive batch, as no reduction)
        0,          # q_batch_idx
        0,          # q_seq_idx
        PAGE_SIZE,        # Let's use the whole cache page
        0           # padding to 32 bytes
    ],
    [OP_STOP,0,0,0,0,0,0,0] # Nop
], dtype=torch.bfloat16)
Stops = torch.zeros_like(Ops) # Just stop instructions
Ops = torch.stack([Ops]+[Stops]*147, dim=0).cuda() # Stack a single Ops and 147 copies of Nops on top of each other to make a tensor
# print(Ops)


# # QK Dimension(s).
# B, R, H, Z, Zc, Zr = 2, 0, 1, 576, 512, 64

# Input Dimension(s).
L = 1
D = 7_168

# Softmax Scale.
Sz = 1 / math.sqrt(D_QK)

# Latent Cache Specific(s).
NUM_PAGES, PAGE_SIZE = 100, 256

# Sequence Length Specific(s).
# T = 32_768 // PAGE_SIZE
T = 1
# Sl, Sh = 32, 1_024

# Initialize Input(s) and Cache(s).
cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()

# query = torch.ones((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()
# query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()

# query growing by H and D_QK
query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()
# query = torch.ones((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()
# linearly scale query by H and D_QK increasing
# query = -1 * query * torch.arange(H, dtype=torch.bfloat16).cuda()[None, None, :, None] - 10.
# query = query * torch.arange(H, dtype=torch.bfloat16).cuda()[None, None, :, None] + 1.
# query = -1 * query * torch.randn((H,), dtype=torch.bfloat16).cuda()[None, None, :, None] + 1.
# query = query * torch.randn((D_QK,), dtype=torch.bfloat16).cuda()[None, None, None, :]

crazy_buffer = torch.randn((100_000,), dtype=torch.bfloat16).cuda()

# query = torch.ones(B, R + 1, H, Z, dtype=DTYPE)

# Define Sequence Length and Block Size(s).

# lengths = torch.randint(Sl, Sh, (B,), dtype=torch.int32)
lengths = torch.full((B,), 256, dtype=torch.int32, device='cuda')
sizes = (lengths + (PAGE_SIZE - 1)).floor_divide(PAGE_SIZE)

total = lengths.sum().item()
maximum = lengths.max().item()


# Allocate Block Table.

table = torch.zeros(B, T, dtype=torch.int32).cuda()

sequence_ids, position_ids = (
    torch.arange(T, dtype=torch.int32, device='cuda')[None, :]
    .expand(B, -1)
    .lt(sizes.view(-1, 1))
    .nonzero(as_tuple=True)
)

table[sequence_ids, position_ids] = (
    torch.randperm(NUM_PAGES, device='cuda')
    [:sequence_ids.size(0)]
    .sort()
    .values
    .int()
)


# Prefill Latent Cache & Extract Padded.

# latent = torch.ones(total, 1, D_QK, dtype=torch.bfloat16).cuda()
latent = torch.randn(total, 1, D_QK, dtype=torch.bfloat16).cuda() * 10
expanded = latent.expand(total, H, D_QK)

sequence_ids, position_ids = (
    torch.arange(maximum, device='cuda')[None, :]
    .expand(B, -1)
    .lt(lengths.view(-1, 1))
    .nonzero(as_tuple=True)
)

padded_key = torch.zeros(B, maximum, H, D_QK, dtype=torch.bfloat16).cuda()
padded_value = torch.zeros(B, maximum, H, D_VO, dtype=torch.bfloat16).cuda()

padded_key[sequence_ids, position_ids] = expanded
padded_value[sequence_ids, position_ids] = expanded[..., :D_VO]

entry_ids = table[sequence_ids, position_ids.floor_divide(PAGE_SIZE)]
column_ids = position_ids.fmod(PAGE_SIZE)
# breakpoint()
cache[entry_ids, column_ids] = latent


# Construct (Optional) Attention Mask.

mask = None

if True:

    bounds = (
        torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :]
        + lengths[:, None]
        - NEW_TOKENS
    )

    mask = (
        torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
        .expand(B, H, NEW_TOKENS, -1)
        .lt(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1))
    )

# &mla_decode_layout::globals::Q,
# Q = torch.ones((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()
# crazy_buffer = torch.randn((100_000,), dtype=torch.bfloat16).cuda()
# &mla_decode_layout::globals::Cache,
# Cache = torch.zeros((100, PAGE_SIZE, D_QK), dtype=torch.bfloat16).cuda() # some big random-ass cache
# &mla_decode_layout::globals::Table,
# Table = torch.randint(0, Cache.shape[0], (B, 1), dtype=torch.int32).cuda() # we can increase this number over time
# for i in range(B):
#     Table[i, 0] = i
# for i in range(B):
#     Cache[Table[i, 0], :] = torch.randn((PAGE_SIZE, D_QK), dtype=torch.bfloat16).cuda()

# &mla_decode_layout::globals::O,
# O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda()
O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda()
dead_buffer = torch.zeros((100_000), dtype=torch.bfloat16).cuda()
# &mla_decode_layout::globals::O_scratch,
O_scratch = torch.zeros((B, 64, D_VO), dtype=torch.float32).cuda()
# &mla_decode_layout::globals::Lvec_scratch,
Lvec_scratch = torch.zeros((B, 64), dtype=torch.float32).cuda()
# &mla_decode_layout::globals::Semaphore,
Semaphore = torch.zeros((B,), dtype=torch.int32).cuda()
# &mla_decode_layout::globals::softmax_scale
softmax_scale = 1.0 / (D_QK ** 0.5)


print('Starting')
print('ops shape', Ops.shape)
print('q shape', query.shape)
print('cache shape', cache.shape)
print('table shape', table.shape)
print('o shape', O.shape)
print('o_scratch shape', O_scratch.shape)
print('lvec_scratch shape', Lvec_scratch.shape)

# breakpoint()

mla_decode.mla_decode(Ops, query.view(B, -1, D_QK), cache.view(NUM_PAGES, -1, D_QK), table, O.view(B, -1, D_VO), O_scratch, Lvec_scratch, Semaphore, softmax_scale)
torch.cuda.synchronize()

# breakpoint()

print('Finished')

print("TK", O)

ref = sdpa(
    query=query.transpose(1, 2),
    key=padded_key.transpose(1, 2),
    value=padded_value.transpose(1, 2),
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
    scale=Sz,
    enable_gqa=False,
).transpose(1, 2)

# breakpoint()

print("ref", ref)

print('ref mean', torch.mean(ref.abs()))

print('TK mean', torch.mean(O.abs()))

print('max abs diff', torch.max(torch.abs(O - ref)))
print('avg abs diff', torch.mean(torch.abs(O - ref)))

breakpoint()

# sizes = (Lengths + (PAGE_SIZE - 1)).floor_divide(PAGE_SIZE).to(device='cuda', dtype=torch.int32)

# total = Lengths.sum().item()
# maximum = Lengths.max().item()

# sequence_ids, position_ids = (
#         torch.arange(MAX_PAGES, dtype=torch.int32, device='cuda')[None, :]
#         .expand(B, -1)
#         .lt(sizes.view(-1, 1))
#         .nonzero(as_tuple=True)
#     )

# Table[sequence_ids, position_ids] = (
#         torch.randperm(CACHE_PAGES, device='cuda')
#         [:sequence_ids.size(0)]
#         .sort()
#         .values
#         .int()
#     )

# # latent = torch.ones(total, 1, D_QK, dtype=torch.bfloat16, device='cuda')
# latent = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device='cuda')
# expanded = latent.expand(total, H, D_QK)

# sequence_ids, position_ids = (
#         torch.arange(maximum, device='cuda')[None, :]
#         .expand(B, -1)
#         .lt(Lengths.view(-1, 1))
#         .nonzero(as_tuple=True)
#     )

# padded_key = torch.zeros(B, maximum, H, D_QK, dtype=torch.bfloat16, device='cuda')
# padded_value = torch.zeros(B, maximum, H, D_VO, dtype=torch.bfloat16, device='cuda')

# padded_key[sequence_ids, position_ids] = expanded
# padded_value[sequence_ids, position_ids] = expanded[..., :D_VO]

# entry_ids = Table[sequence_ids, position_ids.floor_divide(PAGE_SIZE)]
# column_ids = position_ids.fmod(PAGE_SIZE)

# Cache[entry_ids, column_ids] = latent.squeeze(1)

# mask = None
# R = NEW_TOKENS - 1
# if True:
#     bounds = (
#         torch.arange(R + 1, dtype=torch.int32, device='cuda')[None, :]
#         + Lengths[:, None]
#         - R
#     )

#     mask = (
#         torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
#         .expand(B, H, R + 1, -1)
#         .lt(bounds[:, None, :, None].expand(B, H, R + 1, 1))
#     )


# padded_o1 = sdpa(
#     query=Q.transpose(1, 2),
#     key=padded_key.transpose(1, 2),
#     value=padded_value.transpose(1, 2),
#     attn_mask=mask,
#     dropout_p=0.0,
#     is_causal=False,
#     scale=Sz,
#     enable_gqa=False,
# )

# O_ref = padded_o1.transpose(1, 2)

# print("Reference output:")
# print(O_ref)
# print(torch.mean(O_ref.abs()))

# print("\nMax difference:", torch.max(torch.abs(O - O_ref)).item())
# print("Mean difference:", torch.mean(torch.abs(O - O_ref)).item())

# print("B, NEW_TOKENS, H, D_QK, D_VO, PAGE_SIZE, MAX_LENGTH")
# print(B, NEW_TOKENS, H, D_QK, D_VO, PAGE_SIZE, MAX_LENGTH)