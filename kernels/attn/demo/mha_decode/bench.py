import mha_decode
import torch
import numpy as np
import math
import heapq
import time
import random

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule, assign_num_processors
from timings import save_gantt_chart
from pprint import pprint
from scheduler_regression import estimate_schedule_length

torch.manual_seed(0)

# Configuration parameters
HEAD_DIM  = 128
PAGE_SIZE = 256
H = 16                # Number of heads
NUM_PAGES      = 1000 # Number of pages in cache
NUM_PROCESSORS = 132  # Number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE
ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):
    """Initialize Q, K_cache, V_cache, Lengths and Table."""
    B = len(seq_lengths)
    Q       = torch.randn(B, NEW_TOKENS, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table   = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')
    return Q, K_cache, V_cache, Lengths, Table

def create_thundermha_arguments(seq_lengths, new_tokens, num_heads):

    
    t0 = time.time()
    # seq_head_lengths, num_processors_list = assign_num_processors(seq_lengths, num_heads, new_tokens, NUM_PROCESSORS)
    # Create a heap of processor times.
    # processor_times = [(0, p) for p in range(NUM_PROCESSORS)]
    # partial_uid, reduction_uid = 0, sum(num_processors_list)
    # scheduled_tasks = []
    # for (seq_l, h, b), num_p in zip(seq_head_lengths, num_processors_list):
    #     # Get num_p processor lowest processor times
    #     times, processors = tuple(list(x) for x in zip(*processor_times[-num_p:]))
    #     times = {p: t for p, t in zip(processors, times)}
    #     # Generate the new backwards schedule
    #     new_tasks, partial_uid, reduction_uid, times = backward_schedule(
    #         processors, b, h, seq_l, list(range(new_tokens)), partial_uid, reduction_uid, times
    #     )
    #     scheduled_tasks.extend(new_tasks)
    #     processor_times[-num_p:] = [(times[p], p) for p in processors]
    #     processor_times.sort(reverse=True)
    chunkings = [768 if length < 8192 else 3072 for length in seq_lengths]
    scheduled_tasks = sample_schedule_generator(new_tokens, num_heads, seq_lengths, chunkings=chunkings, num_processors=NUM_PROCESSORS)
    t1 = time.time()
    print(f"Time taken to create schedule: {(t1-t0)*1000:.3f} ms")
    
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, num_heads=num_heads, enable_timings=ENABLE_TIMINGS
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    """Runs a single call to the thunder mha decode."""
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    torch.cuda.synchronize()
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    torch.cuda.synchronize()
    return O

def profile_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=10):
    """Profiles the thunder mha decode kernel over a number of iterations."""
    Semaphore.zero_()
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Warm-up call
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(ITERS):
        mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it % 2, Timings)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / ITERS

def run_benchmark_tk(seq_lengths: List[int], new_tokens: int, iterations: int = 10):
    """
    Runs a benchmark on tk mha decoding with the provided sequence lengths (for the cache) and new_tokens (query length).
    This mirrors the flash-attn benchmark configuration.
    """
    print(f"\n----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} -----------")
    torch.manual_seed(0)
    
    # Initialize arguments
    Q, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, new_tokens)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermha_arguments(seq_lengths, new_tokens, H)
    
    # Profile the thunder mha decode kernel
    avg_time = profile_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=iterations)
    print(f"Profiling: Average time per iteration = {avg_time*1e6:.1f} µs over {iterations} iterations")
    
    # save_gantt_chart(Timings, Instructions)
    
    # Compute memory I/O and FLOPS (using similar estimates as in flash-attn)
    total_length = sum(seq_lengths)
    # Memory I/O in bytes: for each token in the cache, H heads, each with key and value (each HEAD_DIM elements) at 2 bytes per element.
    mem_io = total_length * H * (HEAD_DIM + HEAD_DIM) * 2
    
    flops = new_tokens * total_length * H * (HEAD_DIM + 2*HEAD_DIM) * 2
    print(f"Time: {avg_time*1e6:.1f} µs, {(mem_io/1e9) / avg_time:.0f} GB/s, {(flops/1e12) / avg_time:.0f} TFLOPS/s")

if __name__ == "__main__":
    # Run benchmarks with the same configurations as the flash-attn benchmark:
    run_benchmark_tk([4641, 45118, 65536, 256], 4)
    run_benchmark_tk([4641, 45118, 65536, 256], 2)
    run_benchmark_tk([4641, 45118, 65536, 256], 1)
    run_benchmark_tk([256, 512, 1024, 512, 256], 1)
    run_benchmark_tk([4000, 4000, 4000, 4000, 200], 1)
    run_benchmark_tk([100, 100, 100, 100, 16000], 1)
