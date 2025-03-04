import mla_decode
import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import (
    sample_schedule_generator,
    priority_schedule_tasks,
    visualize_schedule,
    create_arguments_from_task_schedule,
)
from timings import save_gantt_chart


def create_page_table(B: int, seq_lengths: List[int], PAGE_SIZE: int, MAX_LENGTH: int, NUM_PAGES: int):
    """
    Create a page table for each sequence.
    MAX_LENGTH is the maximum allowed sequence length.
    Each batch element i has length seq_lengths[i].
    """
    # T is the number of pages available per sequence (based on MAX_LENGTH)
    T = MAX_LENGTH // PAGE_SIZE  
    table_tensor = torch.zeros(B, T, dtype=torch.int32).cuda()
    lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    sizes = (lengths_tensor + (PAGE_SIZE - 1)) // PAGE_SIZE

    # For each batch element, fill only the first sizes[i] pages.
    for i in range(B):
        num_pages_needed = int(sizes[i].item())
        randperm = torch.randperm(NUM_PAGES, device='cuda')[:num_pages_needed].sort().values.int()
        table_tensor[i, :num_pages_needed] = randperm

    # Compute valid page indices for each sequence.
    sequence_ids, pos_ids = (
        torch.arange(T, dtype=torch.int32, device='cuda')[None, :].expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )
    return table_tensor, lengths_tensor, sequence_ids, pos_ids


def schedule_tasks(seq_lengths: List[int], NUM_PROCESSORS: int, NEW_TOKENS: int):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    num_processors = [math.floor(s / sum(seq_lengths) * NUM_PROCESSORS) for s in seq_lengths]
    while sum(num_processors) < NUM_PROCESSORS:
        min_idx = num_processors.index(min(num_processors))
        num_processors[min_idx] += 1
    while min(num_processors) < 4:
        max_idx = num_processors.index(max(num_processors))
        min_idx = num_processors.index(min(num_processors))
        num_processors[max_idx] -= 1
        num_processors[min_idx] += 1

    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(NEW_TOKENS)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    
    visualize_schedule(scheduled_tasks, num_processors=NUM_PROCESSORS)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS
    )
    
    return scheduled_tasks, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings


def initialize_data(B: int, seq_lengths: List[int], D_QK: int, D_VO: int, H: int,
                    NUM_PAGES: int, PAGE_SIZE: int, table_tensor: torch.Tensor):
    """
    Initialize latent data and cache per sequence.
    Instead of allocating one global tensor using L = max(seq_lengths) and expanding it,
    we allocate per batch element and then pad.
    """
    device = torch.device('cuda')
    L_max = max(seq_lengths)
    
    # Create latent data for each batch element
    latent_list = []
    for i, L in enumerate(seq_lengths):
        latent_i = torch.randn((L, 1, D_QK), dtype=torch.bfloat16, device=device) * 10
        latent_list.append(latent_i)
        
    # Pad the per-element latent data into a global tensor of shape (B, L_max, 1, D_QK)
    latent = torch.zeros((B, L_max, 1, D_QK), dtype=torch.bfloat16, device=device)
    for i, L in enumerate(seq_lengths):
        latent[i, :L, 0, :] = latent_list[i][:, 0, :]
        
    # Create padded key and value tensors by expanding each per–sequence latent along the head dimension.
    padded_key = torch.zeros((B, L_max, H, D_QK), dtype=torch.bfloat16, device=device)
    padded_value = torch.zeros((B, L_max, H, D_VO), dtype=torch.bfloat16, device=device)
    for i, L in enumerate(seq_lengths):
        # Expand latent_i from shape (L, 1, D_QK) to (L, H, D_QK)
        expanded_i = latent_list[i].expand(L, H, D_QK)
        padded_key[i, :L, :, :] = expanded_i
        padded_value[i, :L, :, :] = expanded_i[..., :D_VO]
        
    # Create valid token indices per batch element.
    lengths_tensor = torch.tensor(seq_lengths, device=device).view(-1, 1)
    seq_ids, pos_ids = (
        torch.arange(L_max, device=device)[None, :].expand(B, -1)
        .lt(lengths_tensor)
        .nonzero(as_tuple=True)
    )
    
    # Allocate cache with shape (B, NUM_PAGES, PAGE_SIZE, D_QK)
    cache = torch.zeros((B, NUM_PAGES, PAGE_SIZE, D_QK), dtype=torch.bfloat16, device=device)
    # For each valid token, compute its page and column indices.
    page_ids = table_tensor[seq_ids, (pos_ids // PAGE_SIZE)]
    col_ids = pos_ids % PAGE_SIZE
    cache[seq_ids, page_ids, col_ids, :] = latent[seq_ids, pos_ids, 0, :]
    
    # Create query and output tensor (common to both TK and SDPA).
    query = torch.randn((B, 1, H, D_QK), dtype=torch.bfloat16, device=device) / math.sqrt(D_QK)
    O = torch.zeros((B, 1, H, D_VO), dtype=torch.bfloat16, device=device).contiguous()
    
    # Also, for convenience, define an "expanded" tensor that is the same as padded_key.
    expanded = padded_key.clone()
    
    return cache, latent, expanded, padded_key, padded_value, seq_ids, pos_ids, query, O


def prepare_query_cache(query: torch.Tensor, cache: torch.Tensor, D_QRot: int, NUM_PAGES: int, PAGE_SIZE: int):
    B, NEW_TOKENS, H, D_Qk = query.shape
    # Cache has shape (B, NUM_PAGES, PAGE_SIZE, D_Qk)
    cache_view = cache.view(B, NUM_PAGES, PAGE_SIZE, D_Qk)
    
    query_rot = query[..., -D_QRot:].contiguous()
    query_v   = query[..., :-D_QRot].contiguous()
    
    K_cache = cache_view[..., -D_QRot:].contiguous()
    V_cache = cache_view[..., :-D_QRot].contiguous()
    
    return query_rot, query_v, K_cache, V_cache


def initialize_data_and_views(B: int, seq_lengths: List[int], D_QK: int, D_VO: int, H: int,
                              NUM_PAGES: int, PAGE_SIZE: int, table_tensor: torch.Tensor,
                              D_QRot: int):
    """
    (A) Initialize the latent data, cache, padded keys/values, query, and output tensor.
    (B) Create the query and cache views used by the TK kernel.
    
    Returns a dictionary containing:
      - 'cache', 'latent', 'expanded', 'padded_key', 'padded_value',
      - 'seq_ids', 'pos_ids', 'query', 'O',
      - 'query_rot', 'query_v', 'K_cache', 'V_cache'
    """
    cache, latent, expanded, padded_key, padded_value, seq_ids, pos_ids, query, O = initialize_data(
        B, seq_lengths, D_QK, D_VO, H, NUM_PAGES, PAGE_SIZE, table_tensor
    )
    
    query_rot, query_v, K_cache, V_cache = prepare_query_cache(query, cache, D_QRot, NUM_PAGES, PAGE_SIZE)
    
    return {
        'cache': cache,
        'latent': latent,
        'expanded': expanded,
        'padded_key': padded_key,
        'padded_value': padded_value,
        'seq_ids': seq_ids,
        'pos_ids': pos_ids,
        'query': query,
        'O': O,
        'query_rot': query_rot,
        'query_v': query_v,
        'K_cache': K_cache,
        'V_cache': V_cache
    }


# --- The following functions call the MLA decode kernel.
# Since the original harness assumed a batch size of 1, here we loop over the batch dimension.
def launch_kernel(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache,
                  Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, flag, Timings):
    print("Launching MLA decode kernel...")
    B = query_rot.shape[0]
    for b in range(B):
        mla_decode_module.mla_decode(
            Instructions,
            query_rot[b:b+1],
            query_v[b:b+1],
            K_cache[b:b+1],
            V_cache[b:b+1],
            Table[b:b+1],
            O[b:b+1],
            O_scratch,
            Lvec_scratch,
            Semaphore,
            softmax_scale,
            flag,
            Timings
        )
        torch.cuda.synchronize()
    print("Kernel execution finished.")


def timing(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache,
           Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings,
           num_warmup=4, num_iters=1000):
    B = query_rot.shape[0]
    # Warmup iterations.
    for i in range(num_warmup):
        for b in range(B):
            mla_decode_module.mla_decode(
                Instructions,
                query_rot[b:b+1],
                query_v[b:b+1],
                K_cache[b:b+1],
                V_cache[b:b+1],
                Table[b:b+1],
                O[b:b+1],
                O_scratch,
                Lvec_scratch,
                Semaphore,
                softmax_scale,
                i % 2,
                Timings
            )
        print(f'Ran warmup kernel #{i}')
        torch.cuda.synchronize()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    print(f"\nTiming kernel over {num_iters} iterations...")
    start_event.record()
    for i in range(num_iters):
        for b in range(B):
            mla_decode_module.mla_decode(
                Instructions,
                query_rot[b:b+1],
                query_v[b:b+1],
                K_cache[b:b+1],
                V_cache[b:b+1],
                Table[b:b+1],
                O[b:b+1],
                O_scratch,
                Lvec_scratch,
                Semaphore,
                softmax_scale,
                i % 2,
                Timings
            )
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / (num_iters * B)
    print(f"Average kernel execution time: {avg_time * 1000:.3f} us (per batch call)")
    print(f"Total time for {num_iters} iterations (over all batches): {elapsed_time * 1000:.3f} us\n")
    return avg_time


def test_correctness(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache, Table,
                     O, O1, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings,
                     query, padded_key, padded_value, seq_ids, pos_ids, expanded,
                     D_VO: int, D_QK: int, seq_lengths: List[int], num_iters: int):
    """
    Use the per-sequence lengths to build the attention mask and check correctness.
    """
    B, NEW_TOKENS, H, _ = query.shape
    lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    L = max(seq_lengths)
    bounds = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths[:, None] - NEW_TOKENS)
    mask = (torch.arange(L, dtype=torch.int32, device='cuda')[None, None, None, :]
            .expand(B, H, NEW_TOKENS, -1)
            .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1)))

    from torch.nn.functional import scaled_dot_product_attention as sdpa

    ref = sdpa(
        query=query.transpose(1, 2).float(),
        key=padded_key.transpose(1, 2).float(),
        value=padded_value.transpose(1, 2).float(),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
        enable_gqa=False,
    ).transpose(1, 2)

    print("Testing correctness...")
    # Run several iterations per batch element.
    for b in range(B):
        for i in range(num_iters * 10):
            mla_decode_module.mla_decode(
                Instructions,
                query_rot[b:b+1],
                query_v[b:b+1],
                K_cache[b:b+1],
                V_cache[b:b+1],
                Table[b:b+1],
                O[b:b+1],
                O_scratch,
                Lvec_scratch,
                Semaphore,
                softmax_scale,
                i % 2,
                Timings
            )
        print(f"Batch {b}:")
        print("ref mean:", torch.mean(ref[b].abs()))
        print("Kernel output mean:", torch.mean(O[b].abs()))
        print("Max absolute diff:", torch.max(torch.abs(O[b] - ref[b])))
        print("Avg absolute diff:", torch.mean(torch.abs(O[b] - ref[b])))
        print("Initial kernel output mean:", torch.mean(O1[b].abs()))
        print("Initial Max absolute diff:", torch.max(torch.abs(O1[b] - ref[b])))
        print("Initial Avg absolute diff:", torch.mean(torch.abs(O1[b] - ref[b])))


def finalize(Timings, Instructions):
    save_gantt_chart(Timings, Instructions)
    breakpoint()


def main(seq_lengths_input=None):
    """
    If a single integer is provided, we treat it as a uniform sequence length.
    Otherwise, seq_lengths_input should be a list of integers.
    """
    D_QK, D_VO, D_QRot = 576, 512, 64
    PAGE_SIZE = 256
    NEW_TOKENS, H = 1, 16  # single token (naive single partial op)
    MAX_LENGTH = 65536     # maximum allowed sequence length (cache capacity)
    NUM_PAGES = 1000       # number of pages in cache
    NUM_PROCESSORS = 132   # number of processors

    if seq_lengths_input is None:
        seq_lengths = [65536]
    elif isinstance(seq_lengths_input, int):
        seq_lengths = [seq_lengths_input]
    else:
        seq_lengths = seq_lengths_input
    # Preserve the order of sequence lengths.
    B = len(seq_lengths)
    _ = max(seq_lengths)  # maximum sequence length (used inside initialize_data)

    torch.manual_seed(0)

    # Create page table using per-sequence lengths.
    table_tensor, lengths_tensor, pt_seq_ids, pt_pos_ids = create_page_table(
        B, seq_lengths, PAGE_SIZE, MAX_LENGTH, NUM_PAGES
    )
    Table = table_tensor 

    # Processor scheduling based on provided seq_lengths.
    scheduled_tasks, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = schedule_tasks(
        seq_lengths, NUM_PROCESSORS, NEW_TOKENS
    )

    # Data initialization (per-batch, padded to max(seq_lengths)).
    data = initialize_data_and_views(
        B, seq_lengths, D_QK, D_VO, H, NUM_PAGES, PAGE_SIZE, table_tensor, D_QRot
    )
    
    query        = data['query']
    O            = data['O']
    padded_key   = data['padded_key']
    padded_value = data['padded_value']
    query_rot    = data['query_rot']
    query_v      = data['query_v']
    K_cache      = data['K_cache']
    V_cache      = data['V_cache']
    cache        = data['cache']
    seq_ids      = data['seq_ids']
    pos_ids      = data['pos_ids']
    expanded     = data['expanded']
    
    softmax_scale = 1.0 / math.sqrt(D_QK)
    
    # Launch kernel (TK)
    launch_kernel(mla_decode, Instructions, query_rot, query_v, K_cache, V_cache,
                  Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings)
    O1 = O.clone()
    
    # timing
    num_warmup = 4
    num_iters = 1000
    timing(mla_decode, Instructions, query_rot, query_v, K_cache, V_cache,
           Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings, num_warmup, num_iters)

    # SDPA reference correctness
    test_correctness(
        mla_decode, Instructions, query_rot, query_v, K_cache, V_cache, 
        Table, O, O1, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings,
        query, padded_key, padded_value, seq_ids, pos_ids, expanded, D_VO, D_QK, seq_lengths, num_iters
    )

    finalize(Timings, Instructions)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        try:
            seq_lengths = [int(x) for x in sys.argv[1].split(',')]
            print(seq_lengths)
        except Exception:
            seq_lengths = int(sys.argv[1])
    else:
        seq_lengths = None
    main(seq_lengths)
