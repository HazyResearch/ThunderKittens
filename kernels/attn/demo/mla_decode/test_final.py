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
# Main test script (integrating the reference page table creation)
# ----------------------------------------------------------------------
def main():
    D_QK, D_VO, D_QRot = 576, 512, 64
    PAGE_SIZE = 256
    B, NEW_TOKENS, H = 1, 4, 16  # single token (naive single partial op)
    MAX_LENGTH = 65536
    LENGTH = 65536                 # sequence length
    NUM_PAGES = 1000             # number of pages in cache
    NUM_PROCESSORS = 132         # number of processors

    torch.manual_seed(0)

    T = MAX_LENGTH // PAGE_SIZE
    table_tensor = torch.zeros(B, T, dtype=torch.int32).cuda()
    lengths = torch.full((B,), LENGTH, dtype=torch.int32, device='cuda')
    sizes = (lengths + (PAGE_SIZE - 1)) // PAGE_SIZE
    sequence_ids, pos_ids = (
        torch.arange(T, dtype=torch.int32, device='cuda')[None, :].expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )
    randperm = torch.randperm(NUM_PAGES, device='cuda')[:(LENGTH+PAGE_SIZE-1)//PAGE_SIZE].sort().values.int()
    table_tensor[sequence_ids, pos_ids] = randperm


    # chunking_4671 = [544]*9
    # chunking_1750 = [448]*4
    # chunking_1701 = [448]*4
    # chunking_45096 = [[288,320,320,320,384,384,384,416,416,416,416,448,448,448,448,448]]*7
    # chunking_45096 = [
    #     x+32*i-3*32 for i, piece in enumerate(chunking_45096) for x in piece
    # ]
    # chunking_45096.extend([min(45096-sum(chunking_45096)-i, 384) for i in range(0, 45096-sum(chunking_45096), 384)])
    CHUNK_SIZE = 416
    chunking_4671 = [CHUNK_SIZE]*math.ceil(4671/CHUNK_SIZE)
    chunking_1750 = [CHUNK_SIZE]*math.ceil(1750/CHUNK_SIZE) # will cut at end
    chunking_1701 = [CHUNK_SIZE]*math.ceil(1701/CHUNK_SIZE) # will cut at end
    chunking_45096 = [CHUNK_SIZE]*math.ceil(45096/CHUNK_SIZE)
    chunkings = [chunking_4671, chunking_45096, chunking_1750, chunking_1701]

    # tasks = sample_schedule_generator(new_tokens=NEW_TOKENS, lengths=[4671, 45096, 1750, 1701], chunkings=chunkings)
    # tasks, _ = generate_backward_tasks(
    #     batch_id=0,
    #     sequence_length=LENGTH,
    #     num_processors=NUM_PROCESSORS,
    #     final_time=None,
    #     starting_id=0,
    #     new_tokens=NEW_TOKENS
    # )
    # scheduled_tasks, _ = backward_schedule(list(range(4)), 0, 1024, [0, 1, 2, 3], 0)
    # seq_lengths = sorted([LENGTH])
    seq_lengths = sorted([4671, 1750, 1701, 45096]*2)
    # Begin crappy heuristic to decide processor assignments.
    num_processors = [math.floor(s/sum(seq_lengths)*NUM_PROCESSORS) for s in seq_lengths]
    while min(num_processors) < 4:
        max_idx = num_processors.index(max(num_processors))
        min_idx = num_processors.index(min(num_processors))
        num_processors[max_idx] -= 1
        num_processors[min_idx] += 1

    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks, partial_uid, reduction_uid = [], 0, len(num_processors)
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(list(range(start_p, start_p+num_p)), batch_id, seq_l, [0, 1, 2, 3], partial_uid, reduction_uid)
        scheduled_tasks.extend(new_tasks)
    # scheduled_tasks, _ = backward_schedule(list(range(start_processors, start_processors+num_processors)), 0, 45096, [0, 1, 2, 3], 0)
    # scheduled_tasks, _ = backward_schedule(list(range(NUM_PROCESSORS)), 0, LENGTH, [0, 1, 2, 3], 0)
    # tasks = sample_schedule_generator(new_tokens=NEW_TOKENS, lengths=[LENGTH], chunkings=[512])
    # tasks = sample_schedule_generator(new_tokens=NEW_TOKENS, lengths=seq_lengths, chunkings=[416,416,416,416])
    # scheduled_tasks = priority_schedule_tasks(tasks, num_processors=NUM_PROCESSORS)
    visualize_schedule(scheduled_tasks, num_processors=NUM_PROCESSORS)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS
    )
    Table = table_tensor

    cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()
    total = LENGTH  # for one sequence
    latent = (torch.randn((total, 1, D_QK), dtype=torch.bfloat16).cuda() * 10)
    
    expanded = latent.expand(total, H, D_QK)
    maximum = LENGTH
    padded_key = torch.zeros((B, maximum, H, D_QK), dtype=torch.bfloat16).cuda()
    padded_value = torch.zeros((B, maximum, H, D_VO), dtype=torch.bfloat16).cuda()
    seq_ids, pos_ids = (
        torch.arange(maximum, device='cuda')[None, :].expand(B, -1)
        .lt(torch.tensor([LENGTH], device='cuda').view(-1, 1))
        .nonzero(as_tuple=True)
    )

    # cache from latent using the page table.
    entry_ids  = table_tensor[seq_ids, pos_ids.floor_divide(PAGE_SIZE)]
    column_ids = pos_ids.fmod(PAGE_SIZE)
    cache[entry_ids, column_ids] = latent # / math.sqrt(D_QK)

    query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda() / math.sqrt(D_QK)

    O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()

    softmax_scale = 1.0 / math.sqrt(D_QK)
    
    cache_view = cache.view(B, NUM_PAGES, PAGE_SIZE, D_QK)
    '''
    Changes to the interface:
    - QRot is now (B, NEW_TOKENS, H, D_QRot)
    - QV is now (B, NEW_TOKENS, H, D_VO)
    - K_cache is now (B, NUM_PAGES, PAGE_SIZE, D_QRot)
    - V_cache is now (B, NUM_PAGES, PAGE_SIZE, D_VO)
    '''
    query_rot = query[..., :D_QRot].contiguous()
    query_v = query[..., D_QRot:].contiguous()
    K_cache = cache_view[..., :D_QRot].contiguous()
    V_cache = cache_view[..., D_QRot:].contiguous()

    print("Launching MLA decode kernel...")
    mla_decode.mla_decode(Instructions, query_rot, query_v,
                          K_cache, V_cache,
                          Table, O, O_scratch, Lvec_scratch, Semaphore,
                          softmax_scale, 1, Timings)
    torch.cuda.synchronize()
    print("Kernel execution finished.")
    print("Kernel output O:\n", O)
    O1 = O.clone()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    print(f"\nTiming kernel over {num_iters} iterations...")
    
    # Warmup
    for _ in range(4):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                            K_cache, V_cache,
                            Table, O, O_scratch, Lvec_scratch, Semaphore,
                            softmax_scale, _%2, Timings)
        print(f'ran warmup kernel #{_}')

        torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iters):
        mla_decode.mla_decode(Instructions, query_rot, query_v,
                            K_cache, V_cache,
                            Table, O, O_scratch, Lvec_scratch, Semaphore,
                            softmax_scale, _%2, Timings)
    
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iters
    print(f"Average kernel execution time: {avg_time*1000:.3f} us")
    print(f"Total time for {num_iters} iterations: {elapsed_time*1000:.3f} us\n")

    bounds = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths[:, None] - NEW_TOKENS)
    mask = (torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
            .expand(B, H, NEW_TOKENS, -1)
            .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1)))

    from torch.nn.functional import scaled_dot_product_attention as sdpa
    
    padded_key[seq_ids, pos_ids] = expanded # / math.sqrt(D_QK)
    padded_value[seq_ids, pos_ids] = expanded[..., :D_VO]#  / math.sqrt(D_QK)
    
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
    
    # print("Reference output:\n", ref)
    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean:", torch.mean(O.abs()))
    print("Max absolute diff:", torch.max(torch.abs(O - ref)))
    print("Avg absolute diff:", torch.mean(torch.abs(O - ref)))
    print("Initial kernel output mean:", torch.mean(O1.abs()))
    print("Initial Max absolute diff:", torch.max(torch.abs(O1 - ref)))
    print("Initial Avg absolute diff:", torch.mean(torch.abs(O1 - ref)))

    save_gantt_chart(Timings, Instructions)

    breakpoint()

if __name__ == '__main__':
    main()
