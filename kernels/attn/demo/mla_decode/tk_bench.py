
"""
Usage:
    python tk_bench.py --batch 32 --tokens_per_batch 4 --iterations 1000
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode Benchmark")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (number of output batches)")
    parser.add_argument("--tokens_per_batch", type=int, default=4, help="New tokens per batch (tasks per batch)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of kernel launches for timing")
    parser.add_argument("--length", type=int, default=128, help="Number of tokens in the cache")
    parser.add_argument("--d_qk", type=int, default=576, help="Dimension for Q/K")
    parser.add_argument("--d_vo", type=int, default=512, help="Dimension for V/O")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--page_size", type=int, default=256, help="Page size (tokens per page in cache)")
    parser.add_argument("--num_rows", type=int, default=32, help="NUM_ROWS parameter")
    parser.add_argument("--max_uid", type=int, default=1024, help="Maximum UID (size for table/scratch buffers)")
    parser.add_argument("--grid", type=int, default=None, help="Grid size (overrides batch*tokens_per_batch if set)")
    args = parser.parse_args()

    B = args.batch
    T = args.tokens_per_batch
    grid = args.grid if args.grid is not None else B * T
    
    num_task_iters = 3  
    # assume a 3–iteration instruction tensor (iter 0: partial op, iter 1 & 2: OP_STOP)

    # opcodes
    OP_STOP = 0
    OP_PARTIAL = 1

    # instructions tensor.
    # instructions type is gl<int, 1, -1, -1, 8>, shape (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")
    base_uid = 1000

    # each task (i.e. each tblock) in grid, fill task iteration 0 with a partial op
    # instruction layout for partial op is:
    #   [opcode, uid, dst.batch_idx, dst.seq_idx, q_batch_idx, q_seq_idx, length, 0]
    #
    # batch mode (dst.batch_idx >= 0) the final results are written to the global O buffer
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

    # task iterations 1 and 2 remain zeros (OP_STOP)

    # print configuration summary
    print("Benchmark configuration:")
    print(f"  Batch size: {B}")
    print(f"  Tokens per batch: {T}")
    print(f"  Grid size (number of tasks): {grid}")
    print(f"  Task iterations: {num_task_iters}")
    print(f"  Cache length: {args.length}")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Page size: {args.page_size}, NUM_ROWS: {args.num_rows}")
    print(f"  Max UID: {args.max_uid}")
    print(f"  Iterations: {args.iterations}")

    # softmax scale factor (used in the kernel)
    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    #----------------------------------------------------------------------------
    # global buffers.
    # Q global: shape (B, T, heads, D_QK)
    q = torch.randn((B, T, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    
    # Cache global: shape (1, num_pages, PAGE_SIZE, D_QK)
    # simplicity we assume the entire cache fits in one page (i.e. args.length <= args.page_size)
    num_pages = 1
    cache = torch.randn((1, num_pages, args.page_size, args.d_qk), dtype=torch.bfloat16, device="cuda")
    # cache with “latent” keys for the first args.length tokens
    latent = torch.randn((args.length, 1, args.d_qk), dtype=torch.bfloat16, device="cuda") * 10
    expanded = latent.expand(args.length, args.heads, args.d_qk)
    padded_key = expanded.unsqueeze(0).clone()  # shape (1, args.length, heads, D_QK)
    cache[0, 0, :args.length, :] = latent.squeeze(1)

    # global: shape (1, 1, max_uid, 1)
    table = torch.zeros((1, 1, args.max_uid, 1), dtype=torch.int32, device="cuda")

    # O global: shape (B, T, heads, D_VO)
    O = torch.empty((B, T, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")

    # O_scratch global: shape (1, max_uid, 64, D_VO)
    O_scratch = torch.empty((1, args.max_uid, 64, args.d_vo), dtype=torch.float32, device="cuda")

    # Lvec_scratch global: shape (1, 1, max_uid, 64)
    Lvec_scratch = torch.empty((1, 1, args.max_uid, 64), dtype=torch.float32, device="cuda")

    # Semaphore global: shape (1, 1, 1, max_uid)
    semaphore = torch.zeros((1, 1, 1, args.max_uid), dtype=torch.int32, device="cuda")

    #----------------------------------------------------------------------------
    # Warm-up the kernel (so any one–time setup cost is amortized)
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
