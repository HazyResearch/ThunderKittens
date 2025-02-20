"""
Usage:
    python tk_bench_reduction.py --num_iters 132 --sm_iters 4 --iterations 1000 --work_iters 4

This script benchmarks the reduction stage of the MLA decode kernel by
launching enough reduction ops to fill the GPU. Each reduction op combines
two source results (each identified by a unique UID) and writes its output
to a unique scratch location (in non–batch mode). The parameter --work_iters
controls how many iterations each reduction op performs, and --sm_iters controls
how many ops each SM should execute.
"""

import torch
import argparse
import math
import mla_decode

def main():
    parser = argparse.ArgumentParser(description="Thunderkittens MLA Decode Reduction Benchmark (Fill GPU)")
    parser.add_argument("--num_iters", type=int, default=132, help="Base number of SMs (reduction ops per SM will be multiplied by --sm_iters)")
    parser.add_argument("--sm_iters", type=int, default=4, help="Number of reduction ops each SM performs")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of kernel launches for timing")
    parser.add_argument("--d_qk", type=int, default=576, help="Dimension for Q/K")
    parser.add_argument("--d_vo", type=int, default=512, help="Dimension for V/O")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads")
    # max_uid should be large enough to cover all source UIDs used.
    parser.add_argument("--max_uid", type=int, default=None, help="Maximum UID (if not provided, computed automatically)")
    args = parser.parse_args()

    # total number of reduction ops (blocks) to launch:
    grid = args.num_iters
    num_task_iters = args.sm_iters
    OP_REDUCTION = 2

    # a base for source UIDs (for example, 1100).
    base_uid = 0
    # For each reduction op, we use two source UIDs:
    #    src_uid0 = base_uid + 2*i
    #    src_uid1 = base_uid + 2*i + 1
    # And we write each result to scratch index = i.
    computed_max = base_uid + 2 * grid
    uid_max = computed_max if args.max_uid is None else args.max_uid

    # instructions tensor of shape (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device="cpu")
    for i in range(grid):
        src_uid0 = base_uid + 2 * i
        src_uid1 = base_uid + 2 * i + 1
        dst_seq = i  
        # each reduction writes its result to a unique scratch index
        
        # layout:
        # [opcode, dst_batch, dst_seq, src_uid0, src_uid1, work_iters, 0, 0]
        instructions[0, i, :, 0] = OP_REDUCTION
        instructions[0, i, :, 1] = 0
        instructions[0, i, :, 2] = -1               # dst_batch < 0: non–batch mode (write to scratch)
        instructions[0, i, :, 3] = dst_seq
        instructions[0, i, :, 4] = src_uid0
        instructions[0, i, :, 5] = src_uid1
        instructions[0, i, :, 6] = src_uid0
        instructions[0, i, :, 7] = src_uid1
    instructions = instructions.cuda()

    print("Reduction benchmark configuration:")
    print(f"  Base SM count (num_iters): {args.num_iters}")
    print(f"  Reduction ops per SM (sm_iters): {args.sm_iters}")
    print(f"  Total reduction ops (blocks): {grid}")
    print(f"  Using source UIDs starting at {base_uid} (two per op)")
    print(f"  D_QK: {args.d_qk}, D_VO: {args.d_vo}, Heads: {args.heads}")
    print(f"  Maximum UID: {uid_max}")
    print(f"  Kernel iterations: {args.iterations}")

    # softmax scale (required by the kernel interface but not used in reduction)
    softmax_scale = 1.0 / math.sqrt(args.d_qk)
    print(f"  Softmax scale: {softmax_scale}")

    # dummy global buffers for inputs not used by reduction.
    q = torch.randn((1, 1, args.heads, args.d_qk), dtype=torch.bfloat16, device="cuda")
    cache = torch.randn((1, 1, 256, args.d_qk), dtype=torch.bfloat16, device="cuda")
    table = torch.zeros((1, 1, uid_max, 1), dtype=torch.int32, device="cuda")
    O = torch.empty((1, 1, args.heads, args.d_vo), dtype=torch.bfloat16, device="cuda")

    # scratch buffers: O_scratch and Lvec_scratch.
    O_scratch = torch.empty((uid_max, 1, 16, args.d_vo), dtype=torch.float32, device="cuda")
    Lvec_scratch = torch.empty((1, uid_max, 1, 16), dtype=torch.float32, device="cuda")
    semaphore = torch.ones((1, 1, 1, uid_max), dtype=torch.int32, device="cuda")

    # # each reduction op, fill scratch buffers for each of the two source UIDs.
    # for i in range(grid):
    #     src_uid0 = base_uid + 2 * i
    #     src_uid1 = base_uid + 2 * i + 1
    #     # each source UID, fill O_scratch and Lvec_scratch for as many iterations as specified.
    #     for uid in [src_uid0, src_uid1]:
    #         for it in range(args.work_iters):
    #             O_scratch[0, uid, it, :] = torch.randn((args.d_vo,), dtype=torch.float32, device="cuda")
    #         Lvec_scratch[0, 0, uid, :] = torch.randn((64,), dtype=torch.float32, device="cuda")
    #         semaphore[0, 0, 0, uid] = 1

    print("Warming up reduction kernel...")
    mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()

    iterations = args.iterations
    start_events = torch.cuda.Event(enable_timing=True)
    end_events   = torch.cuda.Event(enable_timing=True)

    print("Starting reduction benchmark...")
    start_events.record()
    for i in range(iterations):
        mla_decode.mla_decode(instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale)
    torch.cuda.synchronize()
    end_events.record()
    elapsed_ms = start_events.elapsed_time(end_events)
    avg_time_us = elapsed_ms * 1e3 / iterations
    print(f"Reduction kernel average execution time: {avg_time_us:.2f} us over {iterations} iterations.")

if __name__ == "__main__":
    main()
