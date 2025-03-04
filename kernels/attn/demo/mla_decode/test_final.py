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
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule
from timings import save_gantt_chart

# ----------------------------------------------------------------------
# Main test script (multi-batch version: each batch has its own sequence length)
# ----------------------------------------------------------------------
def main(seq_lengths: List[int]):
    # Model and kernel parameters
    D_QK, D_VO, D_QRot = 576, 512, 64
    PAGE_SIZE = 256
    NEW_TOKENS, H = 1, 16   # one new token per batch, H heads
    NUM_PAGES = 1000        # number of pages in cache
    NUM_PROCESSORS = 132    # number of processors

    B = len(seq_lengths)  # batch size equals number of sequences provided
    max_length = max(seq_lengths)  # maximum sequence length across batches

    torch.manual_seed(0)

    # Create a page table for each batch.
    # T is the maximum number of pages needed given max_length.
    T = (max_length + PAGE_SIZE - 1) // PAGE_SIZE
    table_tensor = torch.zeros(B, T, dtype=torch.int32).cuda()
    lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    sizes = (lengths_tensor + (PAGE_SIZE - 1)) // PAGE_SIZE

    # For each batch, fill the table_tensor with a sorted random permutation of page indices.
    for b in range(B):
        num_pages_needed = sizes[b].item()
        randperm = torch.randperm(NUM_PAGES, device='cuda')[:num_pages_needed].sort().values.int()
        table_tensor[b, :num_pages_needed] = randperm

    # --- Scheduler setup (assign processors for each batch) ---
    # Use the raw sequence lengths (one per batch) for processor allocation.
    sched_seq_lengths = seq_lengths[:]  # keep original order
    total_seq = sum(sched_seq_lengths)
    num_processors_list = [math.floor(s / total_seq * NUM_PROCESSORS) for s in sched_seq_lengths]
    while min(num_processors_list) < 4:
        max_idx = num_processors_list.index(max(num_processors_list))
        min_idx = num_processors_list.index(min(num_processors_list))
        num_processors_list[max_idx] -= 1
        num_processors_list[min_idx] += 1

    start_processors = [sum(num_processors_list[:i]) for i in range(len(num_processors_list))]
    scheduled_tasks, partial_uid, reduction_uid = [], 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(sched_seq_lengths, start_processors, num_processors_list)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(NEW_TOKENS)),
            partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)

    visualize_schedule(scheduled_tasks, num_processors=NUM_PROCESSORS)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS
    )
    Table = table_tensor

    # --- Cache and latent tensor initialization ---
    cache = torch.zeros((1, NUM_PAGES, PAGE_SIZE, D_QK), dtype=torch.bfloat16).cuda()

    # Create a latent tensor for each batch and fill the cache using the corresponding page table.
    latent_list = []
    for b in range(B):
        l = seq_lengths[b]
        latent = (torch.randn((l, 1, D_QK), dtype=torch.bfloat16, device='cuda') * 10)
        latent_list.append(latent)
        # Compute per-token indices for this batch.
        positions = torch.arange(l, dtype=torch.int32, device='cuda')
        page_indices = positions // PAGE_SIZE
        col_indices = positions % PAGE_SIZE
        # Lookup the random page for each token from the table.
        pages = table_tensor[b, page_indices]
        # Note: assign the latent (squeezing out the singleton dimension) so that shape matches.
        cache[0, pages, col_indices] = latent[:, 0, :]

    # --- Query and output tensor ---
    query = (torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda() /
             math.sqrt(D_QK))
    O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()
    softmax_scale = 1.0 / math.sqrt(D_QK)
    
    # Prepare cache view and split the query into rotated and non-rotated parts.
    cache_view = cache  # already in (B, NUM_PAGES, PAGE_SIZE, D_QK)
    query_rot = query[..., -D_QRot:].contiguous()
    query_v = query[..., :-D_QRot].contiguous()
    K_cache = cache_view[..., -D_QRot:].contiguous()
    V_cache = cache_view[..., :-D_QRot].contiguous()

    print("Launching MLA decode kernel...")
    mla_decode.mla_decode(Instructions, query_rot, query_v,
                          K_cache, V_cache,
                          Table, O, O_scratch, Lvec_scratch, Semaphore,
                          softmax_scale, 1, Timings)
    torch.cuda.synchronize()
    print("Kernel execution finished.")
    O1 = O.clone()

    # --- Timing the kernel ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    print(f"\nTiming kernel over {num_iters} iterations...")
    
    # Warmup iterations
    for i in range(4):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                              K_cache, V_cache,
                              Table, O, O_scratch, Lvec_scratch, Semaphore,
                              softmax_scale, i % 2, Timings)
        print(f'ran warmup kernel #{i}')
        torch.cuda.synchronize()
    start_event.record()
    
    for i in range(num_iters):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                              K_cache, V_cache,
                              Table, O, O_scratch, Lvec_scratch, Semaphore,
                              softmax_scale, i % 2, Timings)
    
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iters
    print(f"Average kernel execution time: {avg_time*1000:.3f} us")
    print(f"Total time for {num_iters} iterations: {elapsed_time*1000:.3f} us\n")

    # --- Reference SDPA kernel for correctness checking ---
    bounds = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths_tensor[:, None] - NEW_TOKENS)
    mask = (torch.arange(max_length, dtype=torch.int32, device='cuda')[None, None, None, :]
            .expand(B, H, NEW_TOKENS, -1)
            .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1)))

        # --- Prepare reference tensors for SDPA ---
    padded_key = torch.zeros((B, max_length, H, D_QK), dtype=torch.bfloat16).cuda()
    padded_value = torch.zeros((B, max_length, H, D_VO), dtype=torch.bfloat16).cuda()

    for b in range(B):
        l = seq_lengths[b]
        expanded = latent_list[b].expand(l, H, D_QK)
        padded_key[b, :l] = expanded
        padded_value[b, :l] = expanded[..., :D_VO]

    from torch.nn.functional import scaled_dot_product_attention as sdpa
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

    print("Testing correctness...")
    diffs = []
    for i in range(num_iters * 10):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                              K_cache, V_cache,
                              Table, O, O_scratch, Lvec_scratch, Semaphore,
                              softmax_scale, i % 2, Timings)
        max_diff_by_batch = [torch.abs(O[b] - ref[b]).max() for b in range(B)]
        mean_diff_by_batch = [torch.abs(O[b] - ref[b]).mean() for b in range(B)]
        diffs.append((max_diff_by_batch, mean_diff_by_batch))

    print("ref mean by batch:", [torch.mean(ref[b].abs()).item() for b in range(B)])
    print("Kernel output mean by batch:", [torch.mean(O[b].abs()).item() for b in range(B)])
    print("Max absolute diff by batch:", [torch.abs(O[b] - ref[b]).max().item() for b in range(B)])
    print("Avg absolute diff by batch:", [torch.abs(O[b] - ref[b]).mean().item() for b in range(B)])
    print("Initial kernel output mean by batch:", [torch.mean(O1[b].abs()).item() for b in range(B)])
    print("Initial Max absolute diff by batch:", [torch.abs(O1[b] - ref[b]).max().item() for b in range(B)])
    print("Initial Avg absolute diff by batch:", [torch.abs(O1[b] - ref[b]).mean().item() for b in range(B)])

    diffs_tensor = torch.tensor(diffs, device='cuda') # (10k, 2, B)
    print("Max diff by batch across iterations:", [torch.max(diffs_tensor[i, 0, b]).item() for b in range(B)])
    print("Avg diff by batch across iterations:", [torch.mean(diffs_tensor[i, 1, b]).item() for b in range(B)])

    save_gantt_chart(Timings, Instructions)

    breakpoint()

if __name__ == '__main__':
    import sys
    # If multiple sequence lengths are provided on the command line, use them.
    # Otherwise, default to a single sequence of length 65536.
    if len(sys.argv) > 1:
        seq_lengths = [int(x) for x in sys.argv[1:]]
    else:
        seq_lengths = [65536]
    main(seq_lengths)
