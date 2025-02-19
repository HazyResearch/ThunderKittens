#!/usr/bin/env python3
"""
Usage:
    python tk_bench.py --batch 32 --tokens_per_batch 4 --iterations 1000

This script benchmarks an end-to-end decode operation by scheduling a
complete group of ops:
  - Task Iteration 0: Four partial ops.
  - Task Iteration 1: Two first-level reductions combining the partial results.
  - Task Iteration 2: One final reduction combining the two intermediate results.
Non–batch mode is used (destination batch index = -1) so that partial results
are written to scratch buffers.
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode End-to-End Benchmark (Multi–Page)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (number of output batches)")
    parser.add_argument("--tokens_per_batch", type=int, default=4, help="New tokens per batch (tasks per batch)")
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

    # For an end-to-end decode group we assume a fixed pattern:
    # - 4 partial ops (iteration 0)
    # - 2 reduction ops (iteration 1)
    # - 1 final reduction op (iteration 2)
    # Other tasks (if any) are treated as no–ops (OP_STOP).
    #
    # (In non–batch mode the destination index is encoded by a negative batch value.)
    #
    # Define opcodes.
    OP_STOP    = 0
    OP_PARTIAL = 1
    OP_REDUCTION = 2

    # We use a 3–iteration instructions tensor.
    num_task_iters = 3
    B = args.batch
    T = args.tokens_per_batch
    # Pack up to 4 tokens per instruction:
    grid = args.grid if args.grid is not None else B * ((T+3)//4)

    # Allocate the instructions tensor with shape (1, grid, num_task_iters, 8).
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")
    # For this end-to-end example we use fixed UIDs:
    # Partial op UIDs.
    uid1 = 100
    uid2 = 101
    uid3 = 102
    uid4 = 103
    # Reduction op UIDs.
    uidA = 200  # intermediate result combining uid1 and uid2
    uidB = 201  # intermediate result combining uid3 and uid4
    # Final reduction writes its output to scratch at index 0.

    # --- Task Iteration 0: Partial Ops ---
    if grid >= 4:
        for i, uid in enumerate([uid1, uid2, uid3, uid4]):
            # Instruction layout:
            # [opcode, uid, dst.batch_idx, dst.seq_idx, q_batch_idx, q_seq_idx, length, 0]
            # Use -1 for dst.batch_idx to indicate non–batch mode (write to scratch).
            instructions[0, i, 0, 0] = OP_PARTIAL
            instructions[0, i, 0, 1] = uid
            instructions[0, i, 0, 2] = -1         # non–batch mode destination
            instructions[0, i, 0, 3] = uid         # use uid as destination sequence index
            instructions[0, i, 0, 4] = 0           # q_batch_idx (dummy)
            instructions[0, i, 0, 5] = 0           # q_seq_idx (dummy)
            instructions[0, i, 0, 6] = args.length # cache length
            instructions[0, i, 0, 7] = 0

    # --- Task Iteration 1: First-Level Reductions ---
    if grid >= 2:
        instructions[0, 0, 1, :] = torch.tensor(
            [OP_REDUCTION, -1, uidA, uid1, uid2, 0, 0, 0],
            dtype=torch.int32, device="cuda")
        instructions[0, 1, 1, :] = torch.tensor(
            [OP_REDUCTION, -1, uidB, uid3, uid4, 0, 0, 0],
            dtype=torch.int32, device="cuda")

    # --- Task Iteration 2: Final Reduction ---
    instructions[0, 0, 2, :] = torch.tensor(
        [OP_REDUCTION, -1, 0, uidA, uidB, 0, 0, 0],
        dtype=torch.int32, device="cuda")

    print("End-to-end decode benchmark configuration:")
    print(f"  Batch size: {B}")
    print(f"  Tokens per batch: {T}")
    print(f"  Grid size (tasks): {grid}")
    print(f"  Task iterations: {num_task_iters}")
    print(f"  Cache length: {args.length}")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Page size: {args.page_size}, NUM_ROWS: {args.num_rows}")
    print(f"  UIDs: partial ops {uid1}, {uid2}, {uid3}, {uid4}; reductions: {uidA}, {uidB}")
    print(f"  Final reduction output written to scratch index 0")
    print(f"  Iterations: {args.iterations}")

    # Softmax scale factor.
    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    #----------------------------------------------------------------------------
    # Global Buffers
    q = torch.randn((B, T, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    
    num_pages = math.ceil(args.length / args.page_size)
    print(f"  Number of cache pages: {num_pages}")
    cache = torch.randn((1, num_pages, args.page_size, args.d_qk),
                        dtype=torch.bfloat16, device="cuda")
    latent = torch.randn((args.length, 1, args.d_qk), dtype=torch.bfloat16, device="cuda") * 10
    for p in range(num_pages):
        start = p * args.page_size
        end = min(args.length, (p + 1) * args.page_size)
        cache[0, p, :end - start, :] = latent[start:end, 0, :]

    expanded = latent.expand(args.length, args.heads, args.d_qk)
    padded_key = expanded.unsqueeze(0).clone()

    uid_max = (1000 + grid) if args.max_uid is None else args.max_uid
    table = torch.zeros((1, 1, uid_max, num_pages), dtype=torch.int32, device="cuda")
    for uid in [uid1, uid2, uid3, uid4, uidA, uidB]:
        for p in range(num_pages):
            table[0, 0, uid, p] = p

    O = torch.empty((B, T, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")
    O_scratch = torch.empty((1, uid_max, 64, args.d_vo), dtype=torch.float32, device="cuda")
    Lvec_scratch = torch.empty((1, 1, uid_max, 64), dtype=torch.float32, device="cuda")
    semaphore = torch.zeros((1, 1, 1, uid_max), dtype=torch.int32, device="cuda")

    print("Warming up the kernel...")
    mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()

    iterations = args.iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    print("Starting end-to-end benchmark...")
    for i in range(iterations):
        start_events[i].record()
        mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
        torch.cuda.synchronize()
        end_events[i].record()
    elapsed_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    avg_time_us = (sum(elapsed_ms) * 1e3) / iterations    
    print(f"Kernel average execution time: {avg_time_us:.2f} us over {iterations} iterations.")

if __name__ == "__main__":
    main()
