#!/usr/bin/env python3
"""
Usage:
    python tk_bench_reduction.py --iterations 1000
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode Reduction Benchmark")
    # For reduction we only need one op.
    parser.add_argument("--iterations", type=int, default=1000, help="Number of kernel launches for timing")
    parser.add_argument("--d_qk", type=int, default=576, help="Dimension for Q/K")
    parser.add_argument("--d_vo", type=int, default=512, help="Dimension for V/O")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--max_uid", type=int, default=1200, help="Maximum UID (should be >= src_uids)")
    args = parser.parse_args()

    # For the reduction benchmark we use a single task (grid = 1) and a single task iteration.
    grid = 1
    num_task_iters = 1

    # Define opcodes.
    OP_REDUCTION = 2

    # Choose source UIDs (that would have been produced by partial ops) and a destination.
    src_uid0 = 1100
    src_uid1 = 1101
    dst_batch = -1   # non–batch mode so that results are written to scratch.
    dst_seq   = 0    # destination index in scratch.
    
    # Instructions tensor (global type gl<int,1,-1,-1,8>) with shape (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cuda")
    instructions[0, 0, 0, 0] = OP_REDUCTION   # opcode = 2
    instructions[0, 0, 0, 1] = dst_batch      # destination batch index (<0 => scratch)
    instructions[0, 0, 0, 2] = dst_seq        # destination sequence index
    instructions[0, 0, 0, 3] = src_uid0       # source UID 0
    instructions[0, 0, 0, 4] = src_uid1       # source UID 1
    # The remaining entries are unused.
    instructions[0, 0, 0, 5] = 0
    instructions[0, 0, 0, 6] = 0
    instructions[0, 0, 0, 7] = 0

    print("Reduction benchmark configuration:")
    print(f"  Reduction op: combine src_uid {src_uid0} and {src_uid1} into dst_seq {dst_seq} (non–batch mode)")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Maximum UID: {args.max_uid}")
    print(f"  Iterations: {args.iterations}")

    # Softmax scale factor (not used by reduction but required by kernel interface)
    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    # ---------------------------
    # Dummy Global Buffers for unused inputs.
    # Q global: shape (1,1,heads,d_qk) -- not used in reduction.
    q = torch.randn((1, 1, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    # Cache global: dummy allocation.
    cache = torch.randn((1, 1, 256, args.d_qk), dtype=torch.bfloat16, device="cuda")
    # Table global: dummy allocation.
    table = torch.zeros((1, 1, args.max_uid, 1), dtype=torch.int32, device="cuda")
    # O global: dummy allocation.
    O = torch.empty((1, 1, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")

    # ---------------------------
    # O_scratch global: shape (1, max_uid, 64, D_VO)
    O_scratch = torch.empty((1, args.max_uid, 64, args.d_vo), dtype=torch.float32, device="cuda")
    # Lvec_scratch global: shape (1, 1, max_uid, 64)
    Lvec_scratch = torch.empty((1, 1, args.max_uid, 64), dtype=torch.float32, device="cuda")
    # Semaphore global: shape (1, 1, 1, max_uid)
    semaphore = torch.zeros((1, 1, 1, args.max_uid), dtype=torch.int32, device="cuda")

    # ---------------------------
    # For reduction, the kernel expects partial results in O_scratch and Lvec_scratch
    # for each source UID and for each iteration (reduction_template sets args.num_iters = 4).
    # We fill these with random data.
    num_reduction_iters = 4
    for uid in [src_uid0, src_uid1]:
        for it in range(num_reduction_iters):
            # Fill the tile (64 rows, D_VO channels) with random FP32 data.
            O_scratch[0, uid, it, :] = torch.randn((args.d_vo,), dtype=torch.float32, device="cuda")
            # For Lvec_scratch, fill a 64–element vector.
            Lvec_scratch[0, 0, uid, :] = torch.randn((64,), dtype=torch.float32, device="cuda")
        # Set the semaphore to 1 so that the reduction op does not spin.
        semaphore[0, 0, 0, uid] = 1

    # ---------------------------
    # Warm-up the reduction kernel.
    print("Warming up reduction kernel...")
    mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()

    # Benchmark loop using CUDA events.
    iterations = args.iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    print("Starting reduction benchmark...")
    for i in range(iterations):
        start_events[i].record()
        mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
        torch.cuda.synchronize()
        end_events[i].record()

    elapsed_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    avg_time_us = (sum(elapsed_ms) * 1e3) / iterations
    print(f"Reduction kernel average execution time: {avg_time_us:.2f} us over {iterations} iterations.")

if __name__ == "__main__":
    main()
