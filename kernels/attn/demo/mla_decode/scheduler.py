import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

m1 = 0.0465
b1 = 10
m2 = 0.366
b2 = 8

# ----------------------------------------------------------------------
# Scheduling code (task generation, scheduling, and argument creation)
# ----------------------------------------------------------------------
@dataclass
class Task:
    uid: int
    batch_id: int              # Which sequence this task belongs to.
    tok_ids: List[int]         # Query indices
    name: str
    task_type: str             # "partial" or "reduction"
    duration: float
    dependencies: List[int] = field(default_factory=list)
    start: float = None
    finish: float = None
    processor: int = None
    args: dict = field(default_factory=dict)
def generate_sequence_tasks(batch_id: int, length: int, chunking,
                            m1: float, b1: float, b2: float,
                            starting_id: int = 0, new_tokens: int = 1) -> Tuple[List[Task], int]:
    """
    Generates tasks for one sequence.
      - Creates PARTIAL tasks that process contiguous chunks.
      - Then builds MERGE tasks (reductions) in a tree-like reduction until one result remains.
    
    PARTIAL cost = m1 * (chunk length) + b1.
    MERGE cost = b2.
    
    Each task is annotated with batch_id.
    Returns a list of tasks for this sequence and the next available task uid.
    """
    tasks = []
    partial_tasks = [[] for _ in range((new_tokens+3)//4)]
    # Create PARTIAL tasks (with no dependencies)
    if type(chunking) == int:
        chunking = [(min(chunking, length - i)) for i in range(0, length, chunking)]
    position = 0
    for chunk_length in chunking:
        end_pos = min(position + chunk_length, length)
        chunk_length = end_pos - position
        duration = m1 * (((chunk_length+31)//32)*32) + b1
        for n in range((new_tokens+3)//4):
            tok_ids = list(range(4*n, min(4*n+4, new_tokens)))
            task = Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Partial_{position}-{end_pos}_tok{tok_ids[0]}-{tok_ids[-1]}",
                task_type="partial",
                duration=duration,
                dependencies=[],
                tok_ids=tok_ids,
                args={"start": position,
                      "end": end_pos,
                      "write_scratch": (chunk_length != length),
                      "length": length}
            )
            tasks.append(task)
            partial_tasks[n].append(task)
            starting_id += 1
        position += chunk_length
    for new_token in range(new_tokens):
        TREE_BASE = 16
        n_idx = new_token // 4
        merge_tasks = []
        for i in range(0, len(partial_tasks[n_idx])-1, TREE_BASE):
            dep = [partial_tasks[n_idx][j].uid for j in range(i, min(i+TREE_BASE, len(partial_tasks[n_idx])))]
            if i+TREE_BASE+1 == len(partial_tasks[n_idx]):
                dep.append(partial_tasks[n_idx][i+TREE_BASE].uid)
            merge_tasks.append(Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Merge_{i//2}_tok{new_token}",
                task_type="reduction",
                duration=b2 + m2*(len(dep)-1),
                dependencies=dep,
                tok_ids=[new_token],
                args={"write_scratch": (i != 0)}
            ))
            starting_id += 1
        tasks.extend(merge_tasks)
        while len(merge_tasks) > 0:
            next_merge_tasks = []
            for i in range(0, len(merge_tasks)-1, TREE_BASE):
                new_dep = [merge_tasks[j].uid for j in range(i+1, min(i+TREE_BASE, len(merge_tasks)))]
                if i+TREE_BASE+1 == len(merge_tasks):
                    new_dep.append(merge_tasks[i+TREE_BASE].uid)
                merge_tasks[i].dependencies.extend(new_dep)
                merge_tasks[i].duration = b2 + m2*(len(merge_tasks[i].dependencies)-1)
                next_merge_tasks.append(merge_tasks[i])
            merge_tasks = next_merge_tasks
    return tasks, starting_id
    
def priority_schedule_tasks(tasks: List[Task], num_processors: int) -> List[Task]:
    """
    Schedules tasks on available processors using a custom heuristic.
    Updates each task's start, finish, and processor.
    """
    tasks_by_id: Dict[int, Task] = {t.uid: t for t in tasks}
    in_degree: Dict[int, int] = {t.uid: len(t.dependencies) for t in tasks}
    dependents: Dict[int, List[int]] = {t.uid: [] for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            dependents[dep].append(t.uid)

    # Compute total work per sequence.
    seq_total_work: Dict[int, float] = {}
    for t in tasks:
        seq_total_work.setdefault(t.batch_id, 0)
        seq_total_work[t.batch_id] += t.duration

    ready = [t for t in tasks if in_degree[t.uid] == 0]
    processors: List[Tuple[float, int]] = [(0, pid) for pid in range(num_processors)]
    heapq.heapify(processors)

    iters=0
    while ready:
        current_time, proc_id = heapq.heappop(processors)
        ready_partial = [t for t in ready if t.task_type == "partial"]
        ready_merge   = [t for t in ready if t.task_type == "reduction"]

        filtered_merge = [t for t in ready_merge if current_time+1 >= tasks_by_id[t.dependencies[0]].finish]
        if filtered_merge:
            filtered_merge.sort(key=lambda t: (-t.duration, t.uid))
            chosen_task = filtered_merge[0]
        elif ready_partial:
            ready_partial.sort(key=lambda t: (-seq_total_work[t.batch_id], t.uid))
            chosen_task = ready_partial[0]
        elif ready_merge:
            ready_merge.sort(key=lambda t: (-t.duration, t.uid))
            chosen_task = ready_merge[0]
        else:
            chosen_task = ready[0]

        ready.remove(chosen_task)
        if chosen_task.task_type == "partial":
            earliest_start = max([0] + [tasks_by_id[dep].finish for dep in chosen_task.dependencies])
            start_time = max(current_time, earliest_start)
            finish_time = start_time + chosen_task.duration
        else:
            start_time = max(current_time, tasks_by_id[chosen_task.dependencies[0]].finish)
            finish_time = max([max(start_time + b2, tasks_by_id[chosen_task.dependencies[i]].finish) + (len(chosen_task.dependencies)-i)*m2
                               for i in range(1, len(chosen_task.dependencies))])
        chosen_task.start = start_time
        chosen_task.finish = finish_time
        chosen_task.processor = proc_id

        heapq.heappush(processors, (finish_time, proc_id))
        for dep_id in dependents[chosen_task.uid]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                ready.append(tasks_by_id[dep_id])
        iters += 1
        print('Num processed: ', iters)
    if any(in_degree[t.uid] != 0 for t in tasks):
        raise ValueError("Cycle detected in task dependencies!")
    return tasks

def create_arguments_from_task_schedule(tasks: List[Task], new_tokens: int, num_processors: int = 1):
    OP_PARTIAL, OP_REDUCTION = 1, 2

    def make_partial_arg(task: Task) -> List[int]:
        return [OP_PARTIAL,
                task.uid,  # uid
                -task.uid-1 if task.args["write_scratch"] else task.batch_id,  # destination (negative means write scratch)
                min(task.tok_ids),  # start token
                task.batch_id,
                min(task.tok_ids),  # duplicate start token
                task.args["start"], task.args["end"], task.args["length"]] + [0]*23

    def make_merge_arg(task: Task) -> List[int]:
        assert(len(task.dependencies) <= 11+16)
        args_list = [OP_REDUCTION,
                     task.uid,  # uid
                     len(task.dependencies)-1,  # number of dependencies minus one
                     -task.uid-1 if task.args["write_scratch"] else task.batch_id,
                     task.tok_ids[0]] + task.dependencies
        return args_list + [0]*(32 - len(args_list))

    num_instructions = max(t.uid for t in tasks) + 1
    processor_tasks = [[] for _ in range(num_processors)]
    for task in tasks:
        processor_tasks[task.processor].append(task)
    for pid in range(num_processors):
        processor_tasks[pid].sort(key=lambda t: t.start)
    print('Final finish time:', max(t.finish for t in tasks))
    print('Max number of dependencies:', max(len(t.dependencies) for t in tasks))
    max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)
    Instructions = torch.zeros((num_processors, max_num_processor_instructions, 32), dtype=torch.int32, device='cpu')
    O_scratch = torch.zeros((num_instructions, new_tokens, 16, 512), dtype=torch.float32, device='cpu')
    L_scratch = torch.zeros((num_instructions, new_tokens, 16), dtype=torch.float32, device='cpu')
    Semaphore = torch.zeros((num_instructions, new_tokens), dtype=torch.int32, device='cpu')
    Timings = torch.zeros((num_processors, max_num_processor_instructions, 64), dtype=torch.int32, device='cpu')
    for pid in range(num_processors):
        for tid, task in enumerate(processor_tasks[pid]):
            if task.task_type == "partial":
                Instructions[pid, tid, :] = torch.tensor(make_partial_arg(task), dtype=torch.int32, device='cpu')
            elif task.task_type == "reduction":
                Instructions[pid, tid, :] = torch.tensor(make_merge_arg(task), dtype=torch.int32, device='cpu')
    if torch.cuda.is_available():
        return (Instructions.cuda(), O_scratch.cuda(),
                L_scratch.cuda(), Semaphore.cuda(),
                Timings.cuda())
    return Instructions, O_scratch, L_scratch, Semaphore, Timings

def sample_schedule_generator( new_tokens: int = 1, lengths: List[int] = None, chunkings = None, table: list = None) -> List[Task]:
    """
    For demonstration, we schedule one sequence (batch 0) with a specified length.
    Using new_tokens=1 yields one partial task (and no merge tasks).
    The page table is passed in dynamically (if not provided, a default is used).
    """
    if chunkings is None:
        chunkings = [256 if length < 8192 else 512 for length in lengths]
    tasks = []
    next_task_id = 0
    # One sequence: (batch_id, length, chunk_pages)
    sequences = [
        (i, length, chunking) for i, (length, chunking) in enumerate(zip(lengths, chunkings))
    ]
    for batch_id, seq_length, chunking in sequences:
        seq_tasks, next_task_id = generate_sequence_tasks(
            new_tokens=new_tokens,
            batch_id=batch_id,
            length=seq_length,
            chunking=chunking,
            m1=m1, b1=b1, b2=b2,
            starting_id=next_task_id
        )
        tasks.extend(seq_tasks)
    return tasks




def visualize_schedule(tasks: List[Task], num_processors: int):
    """
    Plots a Gantt chart of the scheduled tasks.
    
    Tasks are grouped by processor; each task is shown as a horizontal bar.
    Colors distinguish between 'partial' (blue) and 'reduction' (red) instructions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        'partial': 'skyblue',
        'reduction': 'salmon'
    }
    proc_tasks: Dict[int, List[Task]] = {i: [] for i in range(num_processors)}
    for task in tasks:
        proc_tasks[task.processor].append(task)
    height = 9
    for proc_id, tasks_on_proc in proc_tasks.items():
        y_coord = proc_id * 10
        for task in tasks_on_proc:
            ax.broken_barh(
                [(task.start, task.finish-task.start)],
                (y_coord, height),
                facecolors=colors.get(task.task_type, 'gray')
            )
            ax.text((task.start + task.finish)/2, y_coord + height/2,
                    task.name, ha='center', va='center', fontsize=8, color='black')
    ax.set_xlabel("Time")
    ax.set_ylabel("Processors")
    ax.set_yticks([i * 10 + height/2 for i in range(num_processors)])
    ax.set_yticklabels([f"Processor {i}" for i in range(num_processors)])
    ax.set_title("Gantt Chart of Scheduled Tasks (Priority Scheduler)")
    plt.tight_layout()
    plt.savefig(f"gantt_{int(time.time())}.png")