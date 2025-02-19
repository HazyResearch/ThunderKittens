"""
Usage:
    python tk_bench.py --batch 132 --tokens_per_batch 4 --sm_iters 1 --iterations 1000 --work_iters 4

This script benchmarks an end-to-end decode operation by scheduling a complete group of ops:
  - Task Iteration 0: 4 partial ops per group.
  - Task Iteration 1: 2 first-level reductions combining the partial results.
  - Task Iteration 2: 1 final reduction combining the two intermediate results.
Non–batch mode is used (destination batch index = -1) so that partial results are written to scratch buffers.

Additional parameters:
  --sm_iters : Number of groups (per SM) to run (i.e. additional iterations per SM)
  --work_iters : Number of work iterations each reduction op performs.
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode End-to-End Benchmark (Multi–Page)")
    parser.add_argument("--batch", type=int, default=132, help="Batch size (typically number of SMs)")
    parser.add_argument("--tokens_per_batch", type=int, default=4, help="New tokens per batch (must be 4 to pack one group)")
    parser.add_argument("--sm_iters", type=int, default=1, help="Number of groups per SM (i.e. additional iterations per SM)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of kernel launches for timing")
    parser.add_argument("--work_iters", type=int, default=4, help="Number of work iterations each reduction op performs")
    parser.add_argument("--length", type=int, default=1024, help="Number of tokens in the cache (across pages)")
    parser.add_argument("--d_qk", type=int, default=576, help="Dimension for Q/K")
    parser.add_argument("--d_vo", type=int, default=512, help="Dimension for V/O")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--page_size", type=int, default=256, help="Page size (tokens per page in cache)")
    parser.add_argument("--num_rows", type=int, default=32, help="NUM_ROWS parameter")
    parser.add_argument("--max_uid", type=int, default=None, help="Maximum UID (if not provided, computed automatically)")
    # We no longer use --grid directly; grid will be computed from groups.
    args = parser.parse_args()

    # Number of groups = (number of SMs) * (sm_iters)
    num_groups = args.batch * args.sm_iters
    # In our design, each group uses 4 instructions (one per partial op).
    grid = num_groups * 4
    num_task_iters = 3  # 3 iterations: partial, first-level reduction, final reduction

    # Define opcodes.
    OP_STOP    = 0
    OP_PARTIAL = 1
    OP_REDUCTION = 2

    # We'll use the following UID base values:
    # For partial ops in each group: starting at 100.
    # For first-level reductions: starting at 200.
    # For final reduction: starting at 300.
    base_partial = 100
    base_redux   = 200
    base_final   = 300

    # Create the instructions tensor of shape (1, grid, num_task_iters, 8) and initialize to OP_STOP.
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")

    # For each group, fill the instructions.
    for g in range(num_groups):
        group_offset = g * 4  # each group occupies 4 consecutive instruction slots.
        # Partial op UIDs for this group:
        p0 = base_partial + group_offset + 0
        p1 = base_partial + group_offset + 1
        p2 = base_partial + group_offset + 2
        p3 = base_partial + group_offset + 3
        # First-level reduction UIDs for this group:
        r0 = base_redux + g * 2 + 0
        r1 = base_redux + g * 2 + 1
        # Final reduction UID for this group:
        f  = base_final + g

        # --- Task Iteration 0: Partial Ops ---
        # Fill the 4 instructions for this group with partial op instructions.
        for i, uid in enumerate([p0, p1, p2, p3]):
            instructions[0, group_offset + i, 0, 0] = OP_PARTIAL
            instructions[0, group_offset + i, 0, 1] = uid
            instructions[0, group_offset + i, 0, 2] = -1    # non-batch mode
            instructions[0, group_offset + i, 0, 3] = uid   # use uid as destination sequence index
            instructions[0, group_offset + i, 0, 4] = 0     # dummy q_batch_idx
            instructions[0, group_offset + i, 0, 5] = 0     # dummy q_seq_idx
            instructions[0, group_offset + i, 0, 6] = args.length
            instructions[0, group_offset + i, 0, 7] = 0

        # --- Task Iteration 1: First-Level Reductions ---
        # For each group, we only use the first 2 instruction slots.
        instructions[0, group_offset + 0, 1, 0] = OP_REDUCTION
        instructions[0, group_offset + 0, 1, 1] = -1
        instructions[0, group_offset + 0, 1, 2] = r0
        instructions[0, group_offset + 0, 1, 3] = p0
        instructions[0, group_offset + 0, 1, 4] = p1
        instructions[0, group_offset + 0, 1, 5] = args.work_iters
        instructions[0, group_offset + 0, 1, 6] = 0
        instructions[0, group_offset + 0, 1, 7] = 0

        instructions[0, group_offset + 1, 1, 0] = OP_REDUCTION
        instructions[0, group_offset + 1, 1, 1] = -1
        instructions[0, group_offset + 1, 1, 2] = r1
        instructions[0, group_offset + 1, 1, 3] = p2
        instructions[0, group_offset + 1, 1, 4] = p3
        instructions[0, group_offset + 1, 1, 5] = args.work_iters
        instructions[0, group_offset + 1, 1, 6] = 0
        instructions[0, group_offset + 1, 1, 7] = 0

        # Leave instructions[0, group_offset+2, 1, :] and [group_offset+3, 1, :] as OP_STOP.

        # --- Task Iteration 2: Final Reduction ---
        # We use only the first instruction slot per group.
        instructions[0, group_offset + 0, 2, 0] = OP_REDUCTION
        instructions[0, group_offset + 0, 2, 1] = -1
        instructions[0, group_offset + 0, 2, 2] = f
        instructions[0, group_offset + 0, 2, 3] = r0
        instructions[0, group_offset + 0, 2, 4] = r1
        instructions[0, group_offset + 0, 2, 5] = args.work_iters
        instructions[0, group_offset + 0, 2, 6] = 0
        instructions[0, group_offset + 0, 2, 7] = 0
        # The remaining instruction slots in iteration 2 of this group remain OP_STOP.

    print("End-to-end decode benchmark configuration:")
    print(f"  Batch (SM count): {args.batch}")
    print(f"  Tokens per batch: {args.tokens_per_batch}")
    print(f"  sm_iters (groups per SM): {args.sm_iters}")
    print(f"  Total groups: {num_groups}")
    print(f"  Grid size (total instructions): {grid}")
    print(f"  Work iterations per reduction op: {args.work_iters}")
    print(f"  Cache length: {args.length}")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Page size: {args.page_size}, NUM_ROWS: {args.num_rows}")
    print(f"  Iterations: {args.iterations}")

    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    #----------------------------------------------------------------------------
    # Global Buffers
    # Q global: shape (B, T, heads, D_QK)
    q = torch.randn((args.batch, args.tokens_per_batch, args.heads, args.d_qk),
                    dtype=torch.bfloat16, device="cuda")
    
    num_pages = math.ceil(args.length / args.page_size)
    print(f"  Number of cache pages: {num_pages}")
    cache = torch.randn((1, num_pages, args.page_size, args.d_qk),
                        dtype=torch.bfloat16, device="cuda")
    latent = torch.randn((args.length, 1, args.d_qk),
                         dtype=torch.bfloat16, device="cuda") * 10
    for p in range(num_pages):
        start = p * args.page_size
        end = min(args.length, (p + 1) * args.page_size)
        cache[0, p, :end - start, :] = latent[start:end, 0, :]

    expanded = latent.expand(args.length, args.heads, args.d_qk)
    padded_key = expanded.unsqueeze(0).clone()

    # We'll compute uid_max so that it covers all partial and reduction UIDs.
    uid_max_candidate = max(base_partial + grid, base_redux + num_groups * 2, base_final + num_groups)
    uid_max = uid_max_candidate if args.max_uid is None else args.max_uid

    # For the table, we map each used UID to its page (we simply assign pages 0,1,2,...)
    table = torch.zeros((1, 1, uid_max, num_pages), dtype=torch.int32, device="cuda")
    # Fill table for all UIDs we used in partials and reductions.
    # For simplicity, we map each UID to page 0,1,2,... sequentially.
    for uid in range(100, uid_max):
        for p in range(num_pages):
            table[0, 0, uid, p] = p

    # O global: shape (B, T, heads, D_VO)
    O = torch.empty((args.batch, args.tokens_per_batch, args.heads, args.d_vo),
                    dtype=torch.bfloat16, device="cuda")
    # O_scratch global: shape (1, uid_max, 64, D_VO)
    O_scratch = torch.empty((1, uid_max, 64, args.d_vo), dtype=torch.float32, device="cuda")
    # Lvec_scratch global: shape (1, 1, uid_max, 64)
    Lvec_scratch = torch.empty((1, 1, uid_max, 64), dtype=torch.float32, device="cuda")
    # Semaphore global: shape (1, 1, 1, uid_max)
    semaphore = torch.zeros((1, 1, 1, uid_max), dtype=torch.int32, device="cuda")

    # For each group, we must prefill the scratch buffers for the reduction ops.
    for g in range(num_groups):
        group_offset = g * 4
        # Partial op UIDs:
        p0 = base_partial + group_offset + 0
        p1 = base_partial + group_offset + 1
        p2 = base_partial + group_offset + 2
        p3 = base_partial + group_offset + 3
        # Intermediate reduction UIDs:
        r0 = base_redux + g * 2 + 0
        r1 = base_redux + g * 2 + 1
        # For each partial UID, fill scratch buffers for reduction (they are used as sources in iteration 1).
        for uid in [p0, p1, p2, p3]:
            for it in range(args.work_iters):
                O_scratch[0, uid, it, :] = torch.randn((args.d_vo,), dtype=torch.float32, device="cuda")
            Lvec_scratch[0, 0, uid, :] = torch.randn((64,), dtype=torch.float32, device="cuda")
            semaphore[0, 0, 0, uid] = 1
        # For each intermediate reduction UID (sources for final reduction)
        for uid in [r0, r1]:
            for it in range(args.work_iters):
                O_scratch[0, uid, it, :] = torch.randn((args.d_vo,), dtype=torch.float32, device="cuda")
            Lvec_scratch[0, 0, uid, :] = torch.randn((64,), dtype=torch.float32, device="cuda")
            semaphore[0, 0, 0, uid] = 1

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
