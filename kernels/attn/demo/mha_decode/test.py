import mha_decode
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
from pprint import pprint
from scheduler_regression import estimate_schedule_length

torch.manual_seed(0)

HEAD_DIM  = 128
PAGE_SIZE = 256
H = 16                # H heads
NUM_PAGES      = 1000 # number of pages in cache
NUM_PROCESSORS = 132  # number of processors

MAX_NUM_PAGES = 65536 // PAGE_SIZE

ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):

    B = len(seq_lengths)

    # Need to initialize Q, K_cache, V_cache, Lengths, Table
    Q       = torch.randn(B,        NEW_TOKENS, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table   = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')
    
    return Q, K_cache, V_cache, Lengths, Table

def create_thundermha_arguments(seq_lengths, new_tokens, num_heads):
    
    seq_head_lengths = sorted([(s, h, b) for b, s in enumerate(seq_lengths) for h in range(num_heads)])
    print(len(seq_head_lengths))
    t0 = time.time()
    processor_assignments = [max(math.floor(s / (sum(seq_lengths)*num_heads) * NUM_PROCESSORS), 1) for s in seq_lengths for _ in range(num_heads)]
    
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1
    processor_assignments = sorted([(estimate_schedule_length(p, new_tokens, shb[0]), p, shb, i) for i, (p, shb) in enumerate(zip(processor_assignments, seq_head_lengths))])
    
    total = sum(p for (_, p, _, _) in processor_assignments)
    if total > NUM_PROCESSORS:
        new_processor_assignments = []
    for (sched, p, shb, i) in processor_assignments:
        new_p = max(1, math.floor(p * NUM_PROCESSORS / total))
        new_processor_assignments.append((sched, new_p, shb, i))
    processor_assignments = new_processor_assignments
    
    # adjust
    while sum(p for (_, p, _, _) in processor_assignments) < NUM_PROCESSORS:
        # increase the smallest assignment
        idx = min(range(len(processor_assignments)), key=lambda j: processor_assignments[j][1])
        sched, p, shb, i = processor_assignments[idx]
        processor_assignments[idx] = (sched, p + 1, shb, i)
    while sum(p for (_, p, _, _) in processor_assignments) > NUM_PROCESSORS:
        # decrease the largest assignment
        idx = max(range(len(processor_assignments)), key=lambda j: processor_assignments[j][1])
        sched, p, shb, i = processor_assignments[idx]
        if p > 1:
            processor_assignments[idx] = (sched, p - 1, shb, i)
        else:
            break

    while len(seq_head_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        if best[1]-1 == 0: break
        new_t0, new_tn1 = estimate_schedule_length(best[1]-1, new_tokens, best[2][0]), estimate_schedule_length(worst[1]+1, new_tokens, worst[2][0])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0]  = (new_t0,   best[1]-1, best[2],  best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments     = sorted(processor_assignments)
        else:
            break
    
    num_processors = [None for _ in seq_head_lengths]
    for _, p, (s, h, b), i in processor_assignments:
        print(p, s//128)
        num_processors[i] = min(p, s//128)
    
    # create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for (seq_l, h, b), start_p, num_p in zip(seq_head_lengths, start_processors, num_processors):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), b, h, seq_l, list(range(new_tokens)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f'Time taken to create schedule: {(t1-t0)*1000} ms')
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, num_heads=num_heads, enable_timings=ENABLE_TIMINGS
    )
    
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    torch.cuda.synchronize()
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    torch.cuda.synchronize()
    
    return O

def run_mha_torch(Q, K_cache, V_cache, Lengths, Table):
    B, q_tokens, H, head_dim = Q.shape
    full_K = K_cache[Table].reshape(B, -1, H, head_dim)
    full_V = V_cache[Table].reshape(B, -1, H, head_dim)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    O = torch.zeros_like(Q)
    for b in range(B):
        l = Lengths[b].item() 
        Q_b = Q[b:b+1].transpose(1, 2)
        K_b = full_K[b, :l].unsqueeze(0).transpose(1, 2)
        V_b = full_V[b, :l].unsqueeze(0).transpose(1, 2)
        mask = torch.ones(Q.shape[1], l, dtype=torch.bool).tril(diagonal=l-Q.shape[1]).to(Q.device)
        O_b = torch.nn.functional.scaled_dot_product_attention(
            Q_b,
            K_b, 
            V_b, 
            attn_mask=mask, 
            dropout_p=0.0, 
            is_causal=False, 
            scale=softmax_scale
        )
        O[b:b+1] = O_b.transpose(1, 2)    
    return O

def main():
    seq_lengths=sorted([48000, 457, 2048, 8549, 2000])
    
    NEW_TOKENS = 16
    Q, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    ref = run_mha_torch(Q, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermha_arguments(seq_lengths, NEW_TOKENS, H)
    O = run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    
    print('O', O)
    print('Ref', ref)
    for b in range(len(seq_lengths)):
        print("ref mean:", torch.mean(ref[b].abs()))
        print("Kernel output mean:", torch.mean(O[b].abs()))
        print("Max absolute diff:", torch.max(torch.abs(O[b] - ref[b])))
        print("Avg absolute diff:", torch.mean(torch.abs(O[b] - ref[b])))
    save_gantt_chart(Timings, Instructions)
    breakpoint()

if __name__ == "__main__":
    main()