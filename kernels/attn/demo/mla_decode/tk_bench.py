#!/usr/bin/env python3
"""
Usage:
    python tk_bench.py --batch 32 --tokens_per_batch 4 --iterations 1000
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode Benchmark (Multi–Page)")
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

    B = args.batch
    T = args.tokens_per_batch
    grid = args.grid if args.grid is not None else B * T

    num_task_iters = 3  
    # Assume a 3–iteration instruction tensor (iter 0: partial op, iter 1 & 2: OP_STOP)

    # Define opcodes.
    OP_STOP = 0
    OP_PARTIAL = 1

    # Instructions tensor:
    # Type is gl<int, 1, -1, -1, 8>, shape (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")
    base_uid = 1000
    # Optionally auto–compute max_uid if not provided.
    uid_max = (base_uid + grid) if args.max_uid is None else args.max_uid

    # For each task (i.e. each tblock) in grid, fill task iteration 0 with a partial op.
    # Instruction layout for a partial op is:
    #   [opcode, uid, dst.batch_idx, dst.seq_idx, q_batch_idx, q_seq_idx, length, 0]
    #
    # In batch mode (dst.batch_idx >= 0) the final results are written to the global O buffer.
    for i in range(grid):
        # DISTRIBUTE tasks evenly over batches.
        batch_idx = i % B         # destination batch index (>=0)
        token_idx = i // B        # token index within the batch
        uid = base_uid + i
        instructions[0, i, 0, 0] = OP_PARTIAL      # opcode = OP_PARTIAL (1)
        instructions[0, i, 0, 1] = uid             # unique uid for this op
        instructions[0, i, 0, 2] = batch_idx        # destination: batch index (>=0 for batch mode)
        instructions[0, i, 0, 3] = token_idx        # destination: sequence (token) index
        instructions[0, i, 0, 4] = batch_idx        # q_batch_idx (from Q global)
        instructions[0, i, 0, 5] = token_idx        # q_seq_idx (from Q global)
        instructions[0, i, 0, 6] = args.length      # number of tokens (cache length)
        instructions[0, i, 0, 7] = 0                # unused

    # Task iterations 1 and 2 remain zeros (OP_STOP).

    # Print configuration summary.
    print("Benchmark configuration:")
    print(f"  Batch size: {B}")
    print(f"  Tokens per batch: {T}")
    print(f"  Grid size (number of tasks): {grid}")
    print(f"  Task iterations: {num_task_iters}")
    print(f"  Cache length: {args.length}")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Page size: {args.page_size}, NUM_ROWS: {args.num_rows}")
    print(f"  UID range: {base_uid} to {uid_max-1}")
    print(f"  Iterations: {args.iterations}")

    # Softmax scale factor (used in the kernel).
    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    #----------------------------------------------------------------------------
    # Global Buffers
    # Q global: shape (B, T, heads, D_QK)
    q = torch.randn((B, T, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    
    # For the cache we now support multiple pages.
    # Compute the number of pages needed to store args.length tokens.
    num_pages = math.ceil(args.length / args.page_size)
    print(f"  Number of cache pages: {num_pages}")

    # Cache global: shape (1, num_pages, page_size, D_QK)
    cache = torch.randn((1, num_pages, args.page_size, args.d_qk), dtype=torch.bfloat16, device="cuda")
    # Generate latent keys for args.length tokens.
    latent = torch.randn((args.length, 1, args.d_qk), dtype=torch.bfloat16, device="cuda") * 10

    # Fill each page of the cache with the corresponding slice of latent keys.
    for p in range(num_pages):
        start = p * args.page_size
        end = min(args.length, (p+1) * args.page_size)
        # Note: latent slice shape: (num_tokens_in_page, 1, d_qk)
        # We squeeze the singleton dimension to match cache slice shape: (num_tokens_in_page, d_qk)
        cache[0, p, :end-start, :] = latent[start:end, 0, :]

    # Optionally build a padded key (not used by the kernel directly) for reference.
    expanded = latent.expand(args.length, args.heads, args.d_qk)
    padded_key = expanded.unsqueeze(0).clone()  # shape (1, args.length, heads, D_QK)

    # Table global: now allocate shape (1, 1, uid_max, num_pages)
    table = torch.zeros((1, 1, uid_max, num_pages), dtype=torch.int32, device="cuda")
    # For each op’s uid, fill its table entry with consecutive page IDs.
    # (The kernel uses: next_page_id = table[uid, iter/ITERS_PER_PAGE])
    for i in range(grid):
        uid = base_uid + i
        for p in range(num_pages):
            table[0, 0, uid, p] = p

    # O global: shape (B, T, heads, D_VO)
    O = torch.empty((B, T, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")

    # O_scratch global: shape (1, uid_max, 64, D_VO)
    O_scratch = torch.empty((1, uid_max, 64, args.d_vo), dtype=torch.float32, device="cuda")

    # Lvec_scratch global: shape (1, 1, uid_max, 64)
    Lvec_scratch = torch.empty((1, 1, uid_max, 64), dtype=torch.float32, device="cuda")

    # Semaphore global: shape (1, 1, 1, uid_max)
    semaphore = torch.zeros((1, 1, 1, uid_max), dtype=torch.int32, device="cuda")

    #----------------------------------------------------------------------------
    # Warm-up the kernel (to amortize any one–time setup cost)
    print("Warming up the kernel...")
    mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()

    #----------------------------------------------------------------------------
    # Benchmark loop using CUDA events.
    iterations = args.iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Starting benchmark...")
    start_event.record()
    for _ in range(iterations):
        mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_us = (elapsed_ms * 1e3) / iterations    

    print(f"Kernel average execution time: {avg_time_us:.2f} us over {iterations} iterations.")

if __name__ == "__main__":
    main()
