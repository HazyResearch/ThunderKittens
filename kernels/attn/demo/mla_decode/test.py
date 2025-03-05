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
from graphviz import Digraph
from scheduler_regression import estimate_schedule_length


torch.manual_seed(0)

D_Main, D_Rot = 512, 64
PAGE_SIZE = 256
H = 16                  # H heads
NUM_PAGES = 1000        # number of pages in cache
NUM_PROCESSORS = 132    # number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE

ENABLE_TIMINGS = True

from graphviz import Digraph
def visualize_dependency_graph(tasks):
    """
    Generate a Graphviz Digraph for the dependency graph of tasks.
    
    Each node is labeled with the task's name, uid, processor, start, and finish times.
    An edge is drawn from each dependency task to the dependent task.
    
    Additionally, all tasks with task_type 'partial' are forced to appear on the bottom row.
    """
    # Map uid to task for dependency verification.
    uid_to_task = {task.uid: task for task in tasks}
    
    dot = Digraph(comment="Task Dependency Graph")
    
    # Create nodes for each task.
    for task in tasks:
        label = (
            f"{task.name}\n"
            f"UID: {task.uid}\n"
            f"Type: {task.task_type}\n"
            f"Proc: {task.processor}\n"
            f"start: {round(task.start, 2) if task.start is not None else 'None'}\n"
            f"finish: {round(task.finish, 2) if task.finish is not None else 'None'}"
        )
        dot.node(str(task.uid), label)
    
    # Create edges for each dependency.
    for task in tasks:
        for dep_uid in task.dependencies:
            # Only draw an edge if the dependency exists.
            if dep_uid in uid_to_task:
                dot.edge(str(dep_uid), str(task.uid))
            else:
                # Flag missing dependencies.
                dot.edge(str(dep_uid), str(task.uid), label="missing")
    
    # Force all partial operations to be rendered on the bottom row.
    partial_op_ids = [str(task.uid) for task in tasks if task.task_type == "partial"]
    if partial_op_ids:
        with dot.subgraph() as s:
            s.attr(rank='sink')
            for uid in partial_op_ids:
                s.node(uid)
    
    return dot

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    QRot    = torch.randn(B, NEW_TOKENS, H, D_Rot, dtype=torch.bfloat16, device='cuda')
    QV      = torch.randn(B, NEW_TOKENS, H, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table

def create_thundermla_arguments(seq_lengths, NEW_TOKENS):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    processor_assignments = [math.floor(s / sum(seq_lengths) * NUM_PROCESSORS) for s in seq_lengths]
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1
    processor_assignments = sorted([(estimate_schedule_length(p, NEW_TOKENS, s), p, s, i) for i, (p, s) in enumerate(zip(processor_assignments, seq_lengths))])
    while len(seq_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        new_t0, new_tn1 = estimate_schedule_length(best[1]-1, NEW_TOKENS, best[2]), estimate_schedule_length(worst[1]+1, NEW_TOKENS, worst[2])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0] = (new_t0, best[1]-1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break
    num_processors = [None for _ in seq_lengths]
    for _, p, s, i in processor_assignments:
        num_processors[i] = min(p, s//128)
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(NEW_TOKENS)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    print('scheduled_tasks', len(scheduled_tasks))
    visualize_schedule(scheduled_tasks, NUM_PROCESSORS)

    for task in scheduled_tasks:
        print(task, len(task.dependencies))
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS, enable_timings=ENABLE_TIMINGS
    )
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    if tic is None:
        Semaphore.zero_()
        tic = 1
    OldO, NewO = torch.zeros_like(QV), torch.zeros_like(QV)
    Q_all = torch.concat([QV, QRot], dim=-1).contiguous()
    KV_all = torch.cat([V_cache, K_cache], dim=-1).contiguous()
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    if Timings is not None:
        OldTimings, NewTimings = Timings.clone(), Timings.clone()
    else:
        OldTimings, NewTimings = None, None
    torch.cuda.synchronize()
    if Timings is not None:
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, NewO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, NewTimings)
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, NewO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, NewTimings)
    else:
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, NewO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, NewO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    Semaphore.zero_()
    tic = 1
    torch.cuda.synchronize()
    if Timings is not None:
        mla_decode.mla_decode_old(Instructions, Q_all, KV_all, Table, OldO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, OldTimings)
        mla_decode.mla_decode_old(Instructions, Q_all, KV_all, Table, OldO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, OldTimings)
    else:
        mla_decode.mla_decode_old(Instructions, Q_all, KV_all, Table, OldO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        mla_decode.mla_decode_old(Instructions, Q_all, KV_all, Table, OldO, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    print('NewO-OldO', NewO-OldO)
    return OldO, NewO, OldTimings, NewTimings

def profile_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=1000):
    Semaphore.zero_()
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(1,ITERS+1):
        if Timings is not None:
            mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2, Timings)
        else:
            mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1-t0) / ITERS

def run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table):
    Q = torch.concat([QRot, QV], dim=-1)
    full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
    full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    O = torch.zeros_like(QV)
    for b, l in enumerate(Lengths):
        # assert Q.shape[1] == 1, "Q must have shape (B, 1, H, D) for the time being."
        mask = torch.ones(Q.shape[1], l, dtype=torch.bool).tril(diagonal=l-Q.shape[1]).to(Q.device)
        O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b:b+1].transpose(1, 2),
            full_K[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            full_V[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            is_causal=False,
            attn_mask=mask,
            scale=softmax_scale
        ).transpose(1, 2)
    return O

def main():
    seq_lengths=sorted([4641,45118,1730,1696])
    NEW_TOKENS = 4
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    ref = run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, NEW_TOKENS)
    OldO, NewO, OldTimings, NewTimings = run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean [New, Old]:", torch.mean(NewO.abs()), torch.mean(OldO.abs()))
    print("Max absolute diff [New, Old]:", torch.max(torch.abs(NewO - ref)), torch.max(torch.abs(OldO - ref)))
    print("Avg absolute diff [New, Old]:", torch.mean(torch.abs(NewO - ref)), torch.mean(torch.abs(OldO - ref)))

    time_per_iter = profile_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    print(f"Time per iter: {time_per_iter*1000} ms")

    breakpoint()
    # save_gantt_chart(OldTimings, Instructions, name='old')
    save_gantt_chart(NewTimings, Instructions, name='new')
    # breakpoint()

if __name__ == "__main__":
    main()