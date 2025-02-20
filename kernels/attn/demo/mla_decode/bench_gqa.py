# import mla_decode
import torch
import math

torch.manual_seed(0)

from torch.nn.functional import linear, scaled_dot_product_attention as sdpa
import flash_attn_3_cuda as fa3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--H', type=int, default=8)
parser.add_argument('--HKV', type=int, default=1)
args = parser.parse_args()

D = 128
PAGE_SIZE = 256

B, NEW_TOKENS, H, HKV = 8, 7, args.H, args.HKV
LENGTH = 5

# Softmax Scale.
Sz = 1 / math.sqrt(D)

# Latent Cache Specific(s).
NUM_PAGES, PAGE_SIZE = 10000, 256

# Sequence Length Specific(s).
T = 131072 // PAGE_SIZE
# T = 1
# Sl, Sh = 32, 1_024

# Initialize Input(s) and Cache(s).
k_cache = torch.zeros((NUM_PAGES, PAGE_SIZE, HKV, D), dtype=torch.bfloat16).cuda()
v_cache = torch.zeros((NUM_PAGES, PAGE_SIZE, HKV, D), dtype=torch.bfloat16).cuda()

# query growing by H and D_QK
query = torch.randn((B, NEW_TOKENS, H, D), dtype=torch.bfloat16).cuda()

crazy_buffer = torch.randn((100_000,), dtype=torch.bfloat16).cuda()

lengths = torch.tensor([4671, 45096, 1750, 1701, 4344, 2021, 1051, 8333], dtype=torch.int32, device='cuda')
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
latent_k = torch.randn(total, HKV, D, dtype=torch.bfloat16).cuda() * 10
latent_v = torch.randn(total, HKV, D, dtype=torch.bfloat16).cuda() * 10
sequence_ids, position_ids = (
    torch.arange(maximum, device='cuda')[None, :]
    .expand(B, -1)
    .lt(lengths.view(-1, 1))
    .nonzero(as_tuple=True)
)

entry_ids = table[sequence_ids, position_ids.floor_divide(PAGE_SIZE)]
column_ids = position_ids.fmod(PAGE_SIZE)
# breakpoint()
k_cache[entry_ids, column_ids] = latent_k
v_cache[entry_ids, column_ids] = latent_v

def prepare_data():
    query_data = query.clone()
    kcache = k_cache.clone()
    vcache = v_cache.clone()
    key = None
    value = None
    qv = None
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
        query_data, kcache, vcache, key, value, qv, cache_seqlens, MAX_CAPACITY, block_table, cache_indices, prefix_seqlens, cos, sin, scales_q, scales_k, scales_v, softmax_scale, causal, left_window, right_window, softcap, interleaved, KV_SPLITS
    )

    return data

def run_fa3(data):
    (query, kcache, vcache, key, value, qv, cache_seqlens, MAX_CAPACITY, block_table, cache_indices, prefix_seqlens, cos, sin, scales_q, scales_k, scales_v, softmax_scale, causal, left_window, right_window, softcap, interleaved, KV_SPLITS) = data

    attn_out, softmax_lse, *_ = fa3.fwd(
        # The Q component.
        query,
        # The cached KV components.
        kcache, vcache,
        # The KV components.
        key, value,
        # Query_v
        qv,
        # The output tensor.
        None,
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

print("B, NEW_TOKENS, H, D, HKV, PAGE_SIZE, NUM_PAGES")
print(B, NEW_TOKENS, H, D, HKV, PAGE_SIZE, NUM_PAGES)
print("Q shape", query.shape)
print("K cache shape", k_cache.shape)
print("V cache shape", v_cache.shape)
print("Lengths", lengths)
print("Raw timings", timings)
print("mean", torch.mean(torch.tensor(timings)).item())
print("std", torch.std(torch.tensor(timings)).item())
