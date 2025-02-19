"""
Usage:
    python tk_bench_partial.py --batch 32 --tokens_per_batch 4 --iterations 1000
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode Partial Benchmark")
    parser.add_argument("--batch", type=int, default=132, help="Batch size (number of output batches)")
    parser.add_argument("--tokens_per_batch", type=int, default=4, help="New tokens per batch")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of kernel launches for timing")
    parser.add_argument("--length", type=int, default=1024, help="Number of tokens in the cache (across pages)")
    parser.add_argument("--d_qk", type=int, default=576, help="Dimension for Q/K")
    parser.add_argument("--d_vo", type=int, default=512, help="Dimension for V/O")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--page_size", type=int, default=256, help="Page size (tokens per page in cache)")
    parser.add_argument("--num_rows", type=int, default=32, help="NUM_ROWS parameter")
    parser.add_argument("--max_uid", type=int, default=None, help="Maximum UID (if not provided, it is computed automatically)")
    parser.add_argument("--grid", type=int, default=None, help="Grid size (overrides batch*tokens_per_batch if set)")
    args = parser.parse_args()

    B = args.batch
    T = args.tokens_per_batch
    grid = args.grid if args.grid is not None else B * ((T+3)//4)

    num_task_iters = 1  
    OP_STOP    = 0
    OP_PARTIAL = 1

    # Instructions tensor shape: (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")
    base_uid = 1000
    uid_max = (base_uid + grid) if args.max_uid is None else args.max_uid

    for i in range(grid):
        batch_idx = i % B
        token_idx = i // B
        uid = base_uid + i
        instructions[0, i, 0, 0] = OP_PARTIAL
        instructions[0, i, 0, 1] = uid
        instructions[0, i, 0, 2] = batch_idx
        instructions[0, i, 0, 3] = token_idx
        instructions[0, i, 0, 4] = batch_idx
        instructions[0, i, 0, 5] = token_idx
        instructions[0, i, 0, 6] = args.length
        instructions[0, i, 0, 7] = 0

    print("Partial benchmark configuration:")
    print(f"  Batch size: {B}")
    print(f"  Tokens per batch: {T}")
    print(f"  Grid size (tasks): {grid}")
    print(f"  Cache length: {args.length}")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Page size: {args.page_size}, NUM_ROWS: {args.num_rows}")
    print(f"  UID range: {base_uid} to {uid_max-1}")
    print(f"  Iterations: {args.iterations}")

    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    q = torch.randn((B, T, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    
    num_pages = math.ceil(args.length / args.page_size)
    print(f"  Number of cache pages: {num_pages}")
    cache = torch.randn((1, num_pages, args.page_size, args.d_qk), dtype=torch.bfloat16, device="cuda")
    latent = torch.randn((args.length, 1, args.d_qk), dtype=torch.bfloat16, device="cuda") * 10
    for p in range(num_pages):
        start = p * args.page_size
        end = min(args.length, (p + 1) * args.page_size)
        cache[0, p, :end - start, :] = latent[start:end, 0, :]

    table = torch.zeros((1, 1, uid_max, num_pages), dtype=torch.int32, device="cuda")
    for i in range(grid):
        uid = base_uid + i
        for p in range(num_pages):
            table[0, 0, uid, p] = p

    O = torch.empty((B, T, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")
    O_scratch = torch.empty((1, uid_max, 64, args.d_vo), dtype=torch.float32, device="cuda")
    Lvec_scratch = torch.empty((1, 1, uid_max, 64), dtype=torch.float32, device="cuda")
    semaphore = torch.zeros((1, 1, 1, uid_max), dtype=torch.int32, device="cuda")

    print("Warming up partial kernel...")
    mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()

    iterations = args.iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    print("Starting partial benchmark...")
    for i in range(iterations):
        start_events[i].record()
        mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
        torch.cuda.synchronize()
        end_events[i].record()
    elapsed_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    avg_time_us = (sum(elapsed_ms) * 1e3) / iterations
    print(f"Partial kernel average execution time: {avg_time_us:.2f} us over {iterations} iterations.")

if __name__ == "__main__":
    main()
