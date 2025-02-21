import mla_decode
import torch
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

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
                            starting_id: int = 0, new_tokens: int = 1,
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
        duration = m1 * (((chunk_length+31)//32)*32) + b1
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
                args={"table": page_table[i//page_size : (i+chunk_length+page_size-1)//page_size],
                      "write_scratch": (chunk_length != length),
                      "length": chunk_length}
            )
            tasks.append(task)
            partial_tasks[n].append(task)
            starting_id += 1

    # Build MERGE tasks (if any partial tasks need to be merged).
    for new_token in range(new_tokens):
        n_idx = new_token // 4
        merge_tasks = []
        for i in range(0, len(partial_tasks[n_idx])-1, 2):
            dep = [partial_tasks[n_idx][i].uid, partial_tasks[n_idx][i+1].uid]
            if i+3 == len(partial_tasks[n_idx]):
                dep.append(partial_tasks[n_idx][i+2].uid)
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
            for i in range(0, len(merge_tasks)-1, 2):
                new_dep = [merge_tasks[i+1].uid]
                if i+3 == len(merge_tasks):
                    new_dep.append(merge_tasks[i+2].uid)
                merge_tasks[i].dependencies.extend(new_dep)
                merge_tasks[i].duration = b2 + m2*(len(merge_tasks[i].dependencies)-1)
                next_merge_tasks.append(merge_tasks[i])
            merge_tasks = next_merge_tasks
    return tasks, starting_id

def priority_schedule_tasks(tasks: List[Task], num_processors: int) -> List[Task]:
    """
    Schedules tasks on available processors using a custom heuristic.
    Updates each task’s start, finish, and processor.
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

    while ready:
        current_time, proc_id = heapq.heappop(processors)
        ready_partial = [t for t in ready if t.task_type == "partial"]
        ready_merge   = [t for t in ready if t.task_type == "reduction"]

        if ready_partial:
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
            finish_time = max([max(current_time + b2, tasks_by_id[chosen_task.dependencies[i]].finish) + (len(chosen_task.dependencies)-i)*m2
                               for i in range(1, len(chosen_task.dependencies))])
        chosen_task.start = start_time
        chosen_task.finish = finish_time
        chosen_task.processor = proc_id

        heapq.heappush(processors, (finish_time, proc_id))
        for dep_id in dependents[chosen_task.uid]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                ready.append(tasks_by_id[dep_id])
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
                task.args["length"]] + [0]*9

    def make_merge_arg(task: Task) -> List[int]:
        args_list = [OP_REDUCTION,
                     task.uid,  # uid
                     len(task.dependencies)-1,  # number of dependencies minus one
                     -task.uid-1 if task.args["write_scratch"] else task.batch_id,
                     task.tok_ids[0]] + task.dependencies
        return args_list + [0]*(16 - len(args_list))

    def make_table_entry(task: Task, max_num_pages: int) -> List[int]:
        entry = task.args["table"] + [0]*(max_num_pages - len(task.args["table"]))
        return entry

    num_instructions = max(t.uid for t in tasks) + 1
    max_num_pages = max(len(t.args["table"]) for t in tasks if t.task_type == "partial")
    processor_tasks = [[] for _ in range(num_processors)]
    for task in tasks:
        processor_tasks[task.processor].append(task)
    for pid in range(num_processors):
        processor_tasks[pid].sort(key=lambda t: t.start)
    max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)
    Instructions = torch.zeros((num_processors, max_num_processor_instructions, 16), dtype=torch.int32)
    Table = torch.zeros((num_instructions, max_num_pages), dtype=torch.int32)
    O_scratch = torch.zeros((num_instructions, new_tokens, 16, 512), dtype=torch.float32)
    L_scratch = torch.zeros((num_instructions, new_tokens, 16), dtype=torch.float32)
    Semaphore = torch.zeros((num_instructions, new_tokens), dtype=torch.int32)
    for pid in range(num_processors):
        for tid, task in enumerate(processor_tasks[pid]):
            if task.task_type == "partial":
                Instructions[pid, tid, :] = torch.tensor(make_partial_arg(task), dtype=torch.int32)
                Table[task.uid, :] = torch.tensor(make_table_entry(task, max_num_pages), dtype=torch.int32)
            elif task.task_type == "reduction":
                Instructions[pid, tid, :] = torch.tensor(make_merge_arg(task), dtype=torch.int32)
    if torch.cuda.is_available():
        return (Instructions.cuda(), Table.cuda(), O_scratch.cuda(),
                L_scratch.cuda(), Semaphore.cuda())
    return Instructions, Table, O_scratch, L_scratch, Semaphore

def sample_schedule_generator(page_size: int = 256, new_tokens: int = 1, length: int = 1) -> List[Task]:
    """
    For demonstration, we schedule one sequence (batch 0) with length 256.
    Using new_tokens=1 yields one partial task (and no merge tasks).
    """
    tasks = []
    next_task_id = 0
    # For the page table, use a simple list.
    table = list(range(1000))
    # One sequence: (batch_id, length, chunk_pages)
    sequences = [
        (0, length, 1),
    ]
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


def main():
    D_QK, D_VO = 576, 512
    PAGE_SIZE = 256
    B, NEW_TOKENS, H = 1, 1, 16  # single token (naive single partial op)
    LENGTH = 5                 # sequence length
    NUM_PAGES = 100              # number of pages in cache

    torch.manual_seed(0)

    # scheduled tasks and build kernel args
    tasks = sample_schedule_generator(page_size=PAGE_SIZE, new_tokens=NEW_TOKENS, length=LENGTH)
    scheduled_tasks = priority_schedule_tasks(tasks, num_processors=1)
    Instructions, Table, O_scratch, Lvec_scratch, Semaphore = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=1
    )
    print("Generated Instructions:\n", Instructions)
    print("Generated Table:\n", Table)

    # latent cache (to be used as the key/value source)
    cache = torch.zeros((NUM_PAGES, PAGE_SIZE, 1, D_QK), dtype=torch.bfloat16).cuda()
    total = LENGTH  # for one sequence
    latent = (torch.randn((total, 1, D_QK), dtype=torch.bfloat16).cuda() * 10)
    
    # for pytorch reference, build padded key/value
    # latent to match query’s head dimension
    expanded = latent.expand(total, H, D_QK)
    maximum = LENGTH
    padded_key = torch.zeros((B, maximum, H, D_QK), dtype=torch.bfloat16).cuda()
    padded_value = torch.zeros((B, maximum, H, D_VO), dtype=torch.bfloat16).cuda()
    seq_ids, pos_ids = (
        torch.arange(maximum, device='cuda')[None, :].expand(B, -1)
        .lt(torch.tensor([LENGTH], device='cuda').view(-1, 1))
        .nonzero(as_tuple=True)
    )
    padded_key[seq_ids, pos_ids] = expanded
    padded_value[seq_ids, pos_ids] = expanded[..., :D_VO]

    cache[0, :LENGTH, 0, :] = latent.view(LENGTH, D_QK)

    query = torch.randn((B, NEW_TOKENS, H, D_QK), dtype=torch.bfloat16).cuda()

    lengths = torch.full((B,), LENGTH, dtype=torch.int32, device='cuda')
    bounds  = (torch.arange(NEW_TOKENS, dtype=torch.int32, device='cuda')[None, :] +
              lengths[:, None] - NEW_TOKENS)
    mask    = (torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
               .expand(B, H, NEW_TOKENS, -1)
               .le(bounds[:, None, :, None].expand(B, H, NEW_TOKENS, 1)))

    O = torch.zeros((B, NEW_TOKENS, H, D_VO), dtype=torch.bfloat16).cuda().contiguous()

    softmax_scale = 1.0 / math.sqrt(D_QK)

    print("Launching MLA decode kernel...")
    mla_decode.mla_decode(Instructions, query,
                          cache.view(B, NUM_PAGES, PAGE_SIZE, D_QK),
                          Table, O, O_scratch, Lvec_scratch, Semaphore,
                          softmax_scale)
    torch.cuda.synchronize()
    print("Kernel execution finished.")
    print("Kernel output O:\n", O)

    ref = torch.nn.functional.scaled_dot_product_attention(
        query=query.transpose(1, 2),
        key=padded_key.transpose(1, 2),
        value=padded_value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
        enable_gqa=False,
    ).transpose(1, 2)
    print("Reference output:\n", ref)
    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean:", torch.mean(O.abs()))
    print("Max absolute diff:", torch.max(torch.abs(O - ref)))
    print("Avg absolute diff:", torch.mean(torch.abs(O - ref)))

if __name__ == '__main__':
    main()
