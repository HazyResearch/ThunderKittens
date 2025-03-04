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
    
    max_length = max(seq_lengths)
    T = MAX_LENGTH // PAGE_SIZE  
    table_tensor = torch.zeros(B, T, dtype=torch.int32).cuda()
    lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    sizes = (lengths_tensor + (PAGE_SIZE - 1)) // PAGE_SIZE

    # For each batch element, only fill in the first sizes[i] pages.
    for i in range(B):
        num_pages_needed = int(sizes[i].item())
        randperm = torch.randperm(NUM_PAGES, device='cuda')[:num_pages_needed].sort().values.int()
        table_tensor[i, :num_pages_needed] = randperm

    # indices corresponding to valid pages for each sequence
    sequence_ids, pos_ids = (
        torch.arange(T, dtype=torch.int32, device='cuda')[None, :].expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )
    return table_tensor, lengths_tensor, sequence_ids, pos_ids


def schedule_tasks(seq_lengths: List[int], NUM_PROCESSORS: int, NEW_TOKENS: int):
    # Processor assignment heuristic
    num_processors = [math.floor(s / sum(seq_lengths) * NUM_PROCESSORS) for s in seq_lengths]
    while min(num_processors) < 4:
        max_idx = num_processors.index(max(num_processors))
        min_idx = num_processors.index(min(num_processors))
        num_processors[max_idx] -= 1
        num_processors[min_idx] += 1

    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, len(num_processors)
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
    Initialize latent data and cache.
    Instead of a uniform LENGTH, we use the maximum of seq_lengths for allocation,
    while using each sequence’s true length for indexing.
    """
    total = max(seq_lengths)
    latent = torch.randn((total, 1, D_QK), dtype=torch.bfloat16).cuda() * 10
    expanded = latent.expand(total, H, D_QK)
    
    # Create per-batch sequence indices (each row valid only up to its own length)
    lengths_tensor = torch.tensor(seq_lengths, device='cuda').view(-1, 1)
    seq_ids, pos_ids = (
        torch.arange(total, device='cuda')[None, :].expand(B, -1)
        .lt(lengths_tensor)
        .nonzero(as_tuple=True)
    )
    
    # Fill cache using the page table mapping.
    cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()
    entry_ids = table_tensor[seq_ids, pos_ids.floor_divide(PAGE_SIZE)]
    column_ids = pos_ids.fmod(PAGE_SIZE)
    cache[entry_ids, column_ids] = latent  # Write latent into cache

    # For the reference (SDPA) kernel
    padded_key   = cache[entry_ids, column_ids].expand(B, total, H, D_QK)
    padded_value = cache[entry_ids, column_ids].expand(B, total, H, D_QK)[..., :D_VO]
    
    # Create query and output tensor (common to both TK and SDPA).
    query = torch.randn((B, 1, H, D_QK), dtype=torch.bfloat16).cuda() / math.sqrt(D_QK)
    O = torch.zeros((B, 1, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()
    
    return cache, latent, expanded, padded_key, padded_value, seq_ids, pos_ids, query, O


def prepare_query_cache(query: torch.Tensor, cache: torch.Tensor, D_QRot: int, NUM_PAGES: int, PAGE_SIZE: int):
    B, NEW_TOKENS, H, D_Qk = query.shape
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


def launch_kernel(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache,
                  Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, flag, Timings):
    print("Launching MLA decode kernel...")
    mla_decode_module.mla_decode(
        Instructions, query_rot, query_v, K_cache, V_cache,
        Table, O, O_scratch, Lvec_scratch, Semaphore,
        softmax_scale, flag, Timings
    )
    torch.cuda.synchronize()
    print("Kernel execution finished.")


def timing(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache,
           Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings,
           num_warmup=4, num_iters=1000):
    for i in range(num_warmup):
        mla_decode_module.mla_decode(
            Instructions, query_rot, query_v, K_cache, V_cache,
            Table, O, O_scratch, Lvec_scratch, Semaphore,
            softmax_scale, i % 2, Timings
        )
        print(f'Ran warmup kernel #{i}')
        torch.cuda.synchronize()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    print(f"\nTiming kernel over {num_iters} iterations...")
    start_event.record()
    for i in range(num_iters):
        mla_decode_module.mla_decode(
            Instructions, query_rot, query_v, K_cache, V_cache,
            Table, O, O_scratch, Lvec_scratch, Semaphore,
            softmax_scale, i % 2, Timings
        )
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iters
    print(f"Average kernel execution time: {avg_time * 1000:.3f} us")
    print(f"Total time for {num_iters} iterations: {elapsed_time * 1000:.3f} us\n")
    return avg_time


def test_correctness(mla_decode_module, Instructions, query_rot, query_v, K_cache, V_cache, Table,
                     O, O1, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings,
                     query, padded_key, padded_value, seq_ids, pos_ids, expanded,
                     D_VO: int, D_QK: int, seq_lengths: List[int], num_iters: int):
    """
    Instead of a uniform LENGTH, use the per-sequence lengths.
    Compute the maximum length among sequences for building the attention mask.
    """
    B, NEW_TOKENS, H, _ = query.shape
    lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    maximum = max(seq_lengths)
    bounds = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths[:, None] - NEW_TOKENS)
    mask = (torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
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
    diffs = []
    for i in range(num_iters * 10):
        mla_decode_module.mla_decode(
            Instructions, query_rot, query_v, K_cache, V_cache,
            Table, O, O_scratch, Lvec_scratch, Semaphore,
            softmax_scale, i % 2, Timings
        )
        max_diff = torch.abs(O - ref).max()
        mean_diff = torch.abs(O - ref).mean()
        diffs.append((max_diff, mean_diff))

    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean:", torch.mean(O.abs()))
    print("Max absolute diff:", torch.max(torch.abs(O - ref)))
    print("Avg absolute diff:", torch.mean(torch.abs(O - ref)))
    print("Initial kernel output mean:", torch.mean(O1.abs()))
    print("Initial Max absolute diff:", torch.max(torch.abs(O1 - ref)))
    print("Initial Avg absolute diff:", torch.mean(torch.abs(O1 - ref)))


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

    B = len(seq_lengths)
    max_length = max(seq_lengths)

    torch.manual_seed(0)

    # page table
    table_tensor, lengths_tensor, pt_seq_ids, pt_pos_ids = create_page_table(B, seq_lengths, PAGE_SIZE, MAX_LENGTH, NUM_PAGES)
    Table = table_tensor 

    # Processor scheduling
    scheduled_tasks, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = schedule_tasks(seq_lengths, NUM_PROCESSORS, NEW_TOKENS)

    # initialization
    data = initialize_data_and_views(B, seq_lengths, D_QK, D_VO, H, NUM_PAGES, PAGE_SIZE, table_tensor, D_QRot)
    
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
    
    # tk kernel
    launch_kernel(mla_decode, Instructions, query_rot, query_v, K_cache, V_cache,
                  Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings)
    O1 = O.clone()
    
    # time
    num_warmup = 4
    num_iters = 1000
    timing(mla_decode, Instructions, query_rot, query_v, K_cache, V_cache,
           Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, Timings, num_warmup, num_iters)

    # SDPA reference
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
        except Exception:
            seq_lengths = int(sys.argv[1])
    else:
        seq_lengths = None
        
    main(seq_lengths)
