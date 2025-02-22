# import mla_decode
import torch
import math

torch.manual_seed(0)

from torch.nn.functional import linear, scaled_dot_product_attention as sdpa
import flash_attn_3_cuda as fa3

OP_STOP, OP_PARTIAL, OP_REDUCTION = 0, 1, 2

D_QK, D_VO = 576, 512
PAGE_SIZE = 256

B, NEW_TOKENS, H = 4, 7, 16
LENGTH = 5

# Input Dimension(s).
L = 1
D = 7_168

# Softmax Scale.
Sz = 1 / math.sqrt(D_QK)

# Latent Cache Specific(s).
NUM_PAGES, PAGE_SIZE = 10000, 256

# Sequence Length Specific(s).
T = 65536 // PAGE_SIZE
# T = 1
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
# lengths = torch.full((B,), LENGTH, dtype=torch.int32, device='cuda')
lengths = torch.tensor([4671, 45096, 1750, 1701], dtype=torch.int32, device='cuda')
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

# breakpoint()

padded_key = torch.zeros(B, maximum, H, D_QK, dtype=torch.bfloat16).cuda()
padded_value = torch.zeros(B, maximum, H, D_VO, dtype=torch.bfloat16).cuda()

cum_length = 0
for i in range(B):
    padded_key[i, :lengths[i], :, :] = expanded[cum_length:cum_length+lengths[i], :, :]
    padded_value[i, :lengths[i], :, :] = expanded[cum_length:cum_length+lengths[i], :, :D_VO]
    cum_length += lengths[i]

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
        .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1))
    )

# print('mask', mask)
# print('padded_key', padded_key.shape)
# print('padded_value', padded_value.shape)
# print('table0', table[0])
# print('cache', cache[table[0],:,0,0])
# print('pk', padded_key[0])
# print('query', query.shape, '\n', query)

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

# print("ref", ref)

print('ref mean', torch.mean(ref.abs()))

def prepare_data():
    query_data = query[..., D_VO:].clone()
    kcache = cache[..., D_VO:].clone()
    vcache = cache[..., :D_VO].clone()
    key = None
    value = None
    query_v = query[..., :D_VO].clone()
    out = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda()
    cache_seqlens = lengths.clone()
    MAX_CAPACITY = NUM_PAGES * PAGE_SIZE
    block_table = table.clone()
    cache_indices = None
    prefix_seqlens = None
    cos = None
    sin = None
    scales_q = None
    scales_k = None
    scales_v = None
    softmax_scale = Sz
    causal = True
    left_window = 0
    right_window = 0
    softcap = 0.
    interleaved = False
    KV_SPLITS = 2

    data = (
        query_data, kcache, vcache, key, value, query_v, out, cache_seqlens, MAX_CAPACITY, block_table, cache_indices, prefix_seqlens, cos, sin, scales_q, scales_k, scales_v, softmax_scale, causal, left_window, right_window, softcap, interleaved, KV_SPLITS
    )

    return data

def run_fa3(data):
    (query, kcache, vcache, key, value, query_v, out, cache_seqlens, MAX_CAPACITY, block_table, cache_indices, prefix_seqlens, cos, sin, scales_q, scales_k, scales_v, softmax_scale, causal, left_window, right_window, softcap, interleaved, KV_SPLITS) = data

    attn_out, softmax_lse, *_ = fa3.fwd(
        # The Q component.
        query,
        # The cached KV components.
        kcache, vcache,
        # The KV components.
        key, value,
        # Qv
        query_v,
        # The output tensor.
        out,
        # The Q, K and new K cumulative sequence lengths.
        None, None, None,
        # The sequence length bounds for Q and K.
        None, cache_seqlens,
        # Maximum sequence lengths.
        MAX_CAPACITY, None,
        # The block table for paged-kv.
        block_table,
        # The KV batch index.
        cache_indices,
        # The left padding for Q and K.
        prefix_seqlens,
        # The cached rotary embedding supports.
        cos, sin,
        # Optional scales for query, key and value.
        scales_q, scales_k, scales_v,
        # The softmax scale in attention.
        softmax_scale,
        # Causal self-attention.
        causal,
        # Sliding window attention and sink specific(s).
        left_window, right_window, 0,
        # Attention logits soft-capping.
        softcap,
        # Non-interleaved rotary embedding.
        interleaved,
        # Heuristic based splitting and GQA packing.
        KV_SPLITS, None,
        # SM margin.
        0,
    )

    return attn_out

attn_out = run_fa3(prepare_data())

# print("attn_out", attn_out)

print('attn_out mean', torch.mean(attn_out.abs()))

# breakpoint()

# benchmark it

import time

timings = []

warmup = 3
data = prepare_data()
for i in range(10 + warmup):
    # data = prepare_data()
    torch.cuda.synchronize()
    start = time.time()
    torch.cuda.synchronize()
    attn_out = run_fa3(data)
    torch.cuda.synchronize()
    end = time.time()
    if i >= warmup:
        timings.append(end - start)

print("B, NEW_TOKENS, H, D_QK, D_VO, PAGE_SIZE, NUM_PAGES")
print(B, NEW_TOKENS, H, D_QK, D_VO, PAGE_SIZE, NUM_PAGES)
print("Q shape", query.shape)
print("Lengths", lengths)
print("Raw timings", timings)
print("mean", torch.mean(torch.tensor(timings)).item())
print("std", torch.std(torch.tensor(timings)).item())

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