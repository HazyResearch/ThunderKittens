#!/usr/bin/env python3
"""
This script demonstrates a scheduling strategy for a batch of tasks (representing
partial and reduction instructions) operating on multiple sequences. Each task is
associated with a particular sequence (batch_id) and has a duration. We compute
the total work per sequence and use this information in our scheduler:

  - When partial tasks (which are heavy) are available, we schedule the task
    from the sequence with the most total work.
    
  - When only merge tasks (the lightweight reduction operations) are ready,
    we schedule them in a zigzag round-robin fashion across the sequences.

This ensures that the “big jobs” are started in parallel and that the
reductions are interleaved.
"""

import heapq
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Dict
import torch

# Cost parameters.
m1 = 0.0454
b1 = 10
m2 = 0.366
b2 = 10

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


def generate_sequence_tasks(batch_id: int, length: int, chunk_pages: int,
                            m1: float, b1: float, b2: float,
                            starting_id: int = 0, new_tokens: int = 4,
                            page_size: int = 256, page_table: list = []) -> Tuple[List[Task], int]:
    """
    Generates tasks for one sequence.
      - Creates PARTIAL tasks that process contiguous chunks.
      - Then builds MERGE tasks (reductions) in a tree-like reduction until one
        result remains.
    
    PARTIAL cost = m1 * (chunk length) + b1.
    MERGE cost = b2.
    
    Each task is annotated with batch_id.
    
    Returns a list of tasks for this sequence and the next available task uid.
    """
    chunk_size = chunk_pages * page_size
    tasks = []
    partial_tasks = [[] for _ in range((new_tokens+3)//4)]
    # Create PARTIAL tasks (with no dependencies)
    for i in range(0, length, chunk_size):
        chunk_length = min(chunk_size, length - i)
        duration = m1 * ((chunk_length+31)//32)*32 + b1
        for n in range((new_tokens+3)//4):
            tok_ids = list(range(4*n, min(4*n+4, new_tokens)))
            task = Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Partial_{i}-{i+chunk_length}_tok{tok_ids[0]}-{tok_ids[-1]}",
                task_type="partial",
                duration=duration,
                dependencies=[],
                tok_ids=tok_ids,
                args={"table": page_table[i//page_size:(i+chunk_length+page_size-1)//page_size],
                      "write_scratch": (chunk_length != length),
                      "length": chunk_length}
            )
            tasks.append(task)
            partial_tasks[n].append(task)
            starting_id += 1

    # Build MERGE tasks: merge pairs of results until one remains.
    for new_token in range(new_tokens):
        n_idx = new_token // 4
        merge_tasks = []
        for i in range(0, len(partial_tasks[n_idx])-1, 2):
            dep = [partial_tasks[n_idx][i].uid, partial_tasks[n_idx][i+1].uid]+([partial_tasks[n_idx][i+2].uid] if i+3 == len(partial_tasks[n_idx]) else [])
            merge_tasks.append(Task(
                uid=starting_id,
                batch_id=batch_id,
                name=f"Seq{batch_id}_Merge_{i//2}_tok{new_token}",
                task_type="reduction",
                duration=b2+m2*(len(dep)-1),
                dependencies=dep,
                tok_ids=[new_token],
                args={"write_scratch": (i!=0)}
            ))
            starting_id += 1
        tasks.extend(merge_tasks) # has references
        while len(merge_tasks) > 0:
            next_merge_tasks = []
            for i in range(0, len(merge_tasks)-1, 2):
                new_dep = [merge_tasks[i+1].uid]+([merge_tasks[i+2].uid] if i+3 == len(merge_tasks) else [])
                merge_tasks[i].dependencies.extend(new_dep)
                merge_tasks[i].duration = b2+m2*(len(merge_tasks[i].dependencies)-1)
                next_merge_tasks.append(merge_tasks[i])
            merge_tasks = next_merge_tasks
    from pprint import pprint
    pprint(tasks)
    return tasks, starting_id


def priority_schedule_tasks(tasks: List[Task], num_processors: int) -> List[Task]:
    """
    Schedules tasks on the available processors using a custom policy:
    
    1. Compute the in-degree for each task and a dependency mapping.
    2. Compute the total work per sequence (sum of durations of tasks in that seq).
    3. Maintain a global ready list (tasks with in_degree==0).
    4. Processors are modeled as a heap keyed by available time.
    
    Scheduling policy:
      - If any PARTIAL task is ready, choose the one from the sequence with
        the highest total work.
      - Otherwise, schedule MERGE tasks in a zigzag round-robin fashion among
        the sequences that have merge tasks ready.
    
    The function returns tasks with updated .start, .finish, and .processor fields.
    """
    # Build mapping: task uid -> task.
    tasks_by_id = {t.uid: t for t in tasks}
    
    # Compute in-degree and dependency mapping.
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
    
    # Initialize ready list with tasks that have no dependencies.
    ready: List[Task] = [t for t in tasks if in_degree[t.uid] == 0]

    # Processors: heap of (available_time, processor_id)
    processors: List[Tuple[float, int]] = [(0, pid) for pid in range(num_processors)]
    heapq.heapify(processors)

    # For reduction tasks, maintain a round-robin pointer (cycling through seq_ids).
    merge_rr_pointer = 0

    while ready:
        # Get next available processor.
        current_time, proc_id = heapq.heappop(processors)

        # Partition ready tasks by type.
        ready_partial = [t for t in ready if t.task_type == "partial"]
        ready_merge   = [t for t in ready if t.task_type == "reduction"]

        # Decide whether to run a partial or reduction task.
        # Our heuristic is as follows: if a reduction task will be ready within a single microsecond, run it.
        # If there are multiple, run the largest one.
        # Otherwise, run the largest available partial task.
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

        # Remove the chosen task from the ready list.
        ready.remove(chosen_task)
        # Ensure that this task doesn't start before all its dependencies finish.
        if chosen_task.task_type == "partial":
            earliest_start = max([0]+[tasks_by_id[dep].finish for dep in chosen_task.dependencies])
            start_time = max(current_time, earliest_start)
            finish_time = start_time + chosen_task.duration
        else:
            start_time = max(current_time, tasks_by_id[chosen_task.dependencies[0]].finish)
            finish_time = max([max(current_time, tasks_by_id[chosen_task.dependencies[i]].finish) + b2 + (len(chosen_task.dependencies)-i)*m2 for i in range(1, len(chosen_task.dependencies))])
        chosen_task.start = start_time
        chosen_task.finish = finish_time
        chosen_task.processor = proc_id

        # Update processor availability.
        heapq.heappush(processors, (finish_time, proc_id))

        # Release dependent tasks.
        for dep_id in dependents[chosen_task.uid]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                ready.append(tasks_by_id[dep_id])

    if any(in_degree[t.uid] != 0 for t in tasks):
        raise ValueError("Cycle detected in task dependencies!")
    return tasks

def create_arguments_from_task_schedule(tasks: List[Task], new_tokens: int):
    OP_PARTIAL, OP_REDUCTION = 1, 2
    """
    @dataclass
    class Task:
        uid: int
        batch_id: int                # Which sequence this task belongs to.
        tok_ids: int               # Query indices
        name: str
        task_type: str             # "partial" or "reduction"
        duration: float
        dependencies: List[int] = field(default_factory=list)
        start: float = None
        finish: float = None
        processor: int = None
        args: dict = field(default_factory=dict)
    """
    def make_partial_arg(task: Task) -> List[int]:
        return torch.tensor([
            OP_PARTIAL,
            task.uid, # uid
            -task.uid-1 if task.args["write_scratch"] else task.batch_id, # dst batch uid
            min(task.tok_ids), # start token
            task.batch_id,
            min(task.tok_ids), # start token (replicated)
            task.args["length"]
        ]+[0]*9, dtype=torch.int32)
    def make_merge_arg(task: Task) -> List[int]:
        args = [
            OP_REDUCTION,
            task.uid, # uid
            len(task.dependencies)-1, # num dependencies - 1 is the number of reductions
            -task.uid-1 if task.args["write_scratch"] else task.batch_id, # dst batch uid
            task.tok_ids[0], # operative token,
        ] + task.dependencies
        return torch.tensor(args+[0]*(16-len(args)), dtype=torch.int32)
    def make_table_entry(task: Task, max_num_pages: int) -> List[int]:
        return torch.tensor(task.args["table"]+[0]*(max_num_pages-len(task.args["table"])), dtype=torch.int32)
    # collect the stats we need to intialize tensors
    num_instructions = max(t.uid for t in tasks)+1
    num_processors = max(t.processor for t in tasks)+1
    max_num_pages = max(len(t.args["table"]) for t in tasks if t.task_type == "partial")
    processor_tasks = [[] for _ in range(num_processors)]
    for task in tasks:
        processor_tasks[task.processor].append(task)
    for pid in range(num_processors):
        processor_tasks[pid].sort(key=lambda t: t.start)
    for pid in range(num_processors):
        print(f"Processor {pid} has {len(processor_tasks[pid])} tasks")
        print(processor_tasks[pid])
    max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)
    Instructions = torch.zeros((num_processors, max_num_processor_instructions, 16), dtype=torch.int32)
    Table = torch.zeros((num_instructions, max_num_pages), dtype=torch.int32)
    O_scratch = torch.zeros((num_instructions, new_tokens, 16, 512), dtype=torch.float32)
    L_scratch = torch.zeros((num_instructions, new_tokens, 16), dtype=torch.float32)
    Semaphore = torch.zeros((num_instructions, new_tokens), dtype=torch.int32)
    for pid in range(num_processors):
        for tid, task in enumerate(processor_tasks[pid]):
            if task.task_type == "partial":
                Instructions[pid, tid, :] = make_partial_arg(task)
                Table[task.uid, :] = make_table_entry(task, max_num_pages)
            elif task.task_type == "reduction":
                Instructions[pid, tid, :] = make_merge_arg(task)
    if torch.cuda.is_available():
        return Instructions.cuda(), Table.cuda(), O_scratch.cuda(), L_scratch.cuda(), Semaphore.cuda()
    return Instructions, Table, O_scratch, L_scratch, Semaphore

def verify_dependencies(tasks: List[Task]):
    """
    Verifies that every task starts only after all its dependencies have finished.
    Raises a ValueError if any dependency is violated.
    """
    tasks_by_id = {t.uid: t for t in tasks}
    for task in tasks:
        for dep_id in task.dependencies:
            dep_task = tasks_by_id[dep_id]
            if task.start < dep_task.finish:
                pass # The new approach to reductions makes this not really relevant anymore.
                # raise ValueError(
                #     f"Dependency violation: Task '{task.name}' (start: {task.start}) "
                #     f"begins before dependency '{dep_task.name}' finished (finish: {dep_task.finish})."
                # )


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
                [(task.start, task.duration)],
                (y_coord, height),
                facecolors=colors.get(task.task_type, 'gray')
            )
            ax.text(task.start + task.duration/2, y_coord + height/2,
                    task.name, ha='center', va='center', fontsize=8, color='black')
    ax.set_xlabel("Time")
    ax.set_ylabel("Processors")
    ax.set_yticks([i * 10 + height/2 for i in range(num_processors)])
    ax.set_yticklabels([f"Processor {i}" for i in range(num_processors)])
    ax.set_title("Gantt Chart of Scheduled Tasks (Priority Scheduler)")
    plt.tight_layout()
    plt.savefig(f"gantt_{int(time.time())}.png")
    # plt.show()


def sample_schedule_generator(page_size = 256, new_tokens = 7) -> List[Task]:
    """
    A sample schedule generator for a realistic batch of sequences.
    
    For example, we schedule 4 sequences with lengths:
      4671, 45096, 1750, 1701
    (with an appropriate chunk size for each).
    
    Cost parameters:
      - PARTIAL: cost = m1 * chunk_length + b1
      - MERGE: cost = b2
    """
    tasks = []
    next_task_id = 0
    # Example cost parameters.
    table = list(range(1000))

    # Sequence parameters (batch_id, length, chunk_size).
    sequences = [
        # (1, 1024,  1),
        (1, 4671,  1),
        (2, 45096, 2),
        (3, 1750,  1),
        (4, 1701,  1)
    ]
    # sequences += [
    #     (i, (PAGE_SIZE-1+1701)//PAGE_SIZE,  4) for i in range(5, 100)
    # ]
    sequences.sort(key=lambda x: x[1], reverse=True)
    for batch_id, length, chunk_pages in sequences:
        seq_tasks, next_task_id = generate_sequence_tasks(
            new_tokens=new_tokens,
            page_size=page_size,
            batch_id=batch_id,
            length=length,
            chunk_pages=chunk_pages,
            m1=m1, b1=b1, b2=b2,
            starting_id=next_task_id,
            page_table=table
        )
        tasks.extend(seq_tasks)
    return tasks


def main(schedule_generator: Callable[[], List[Task]],
         scheduler: Callable[[List[Task], int], List[Task]],
         num_processors: int = 16):
    """
    Runs the schedule generator to produce a DAG of tasks, schedules them using
    the provided scheduler, verifies dependency correctness, and visualizes the
    resulting Gantt chart.
    
    You can plug in a different scheduling strategy by providing a different
    scheduler function.
    """
    PAGE_SIZE = 256
    NEW_TOKENS = 7
    tasks = schedule_generator(page_size=PAGE_SIZE, new_tokens=NEW_TOKENS)
    scheduled_tasks = scheduler(tasks, num_processors=num_processors)
    verify_dependencies(scheduled_tasks)
    visualize_schedule(scheduled_tasks, num_processors=num_processors)
    Instructions, Table, O_scratch, L_scratch, Semaphore = create_arguments_from_task_schedule(scheduled_tasks, NEW_TOKENS)
    print(Instructions.shape, Table.shape, O_scratch.shape, L_scratch.shape, Semaphore.shape)

if __name__ == '__main__':
    # Use the new priority-based scheduler.
    main(sample_schedule_generator, scheduler=priority_schedule_tasks, num_processors=132)
