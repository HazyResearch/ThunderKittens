import math
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from mla_decode import __get_quality__

# Timing constants (in microseconds)
PARTIAL_STARTUP_TIME = 3.0         # Startup time for partial operations
PARTIAL_WRITEOUT_TIME = 4.5        # Writeout time for partial operations
PARTIAL_COST_PER_STEP = 1.49       # Cost per step (per 32 tokens) for partial operations
PARTIAL_OVERHEAD = PARTIAL_STARTUP_TIME + PARTIAL_WRITEOUT_TIME # Total overhead for a partial operation.

REDUCTION_STARTUP_TIME = 4.0       # Startup time for reduction operations
# REDUCTION_STARTUP_TIME = 2.0       # Startup time for reduction operations
REDUCTION_WRITEOUT_TIME = 1.0      # Writeout time for reduction operations
REDUCTION_PRODUCER_LATENCY = 1.0   # Latency between a producer load and when the consumer can access it.
REDUCTION_COST_PER_STEP = 0.4      # Cost per reduction step

SYNCHRONIZATION_COST = 0.5         # Synchronization cost between dependent operations


@dataclass
class Task:
    uid: int
    batch_id: int              # Which sequence this task belongs to.
    tok_ids: List[int]         # Query indices
    name: str
    task_type: str             # "partial" or "reduction"
    dependencies: List[int] = field(default_factory=list)
    next_input_time: float = None
    start: float = None
    finish: float = None
    processor: int = None
    args: dict = field(default_factory=dict)
    def __lt__(self, other):
        return self.uid < other.uid # just need a way to break ties.

def backward_schedule(processors: List[int], batch_id: int, seq_length: int, tok_ids: List[int], partial_uid: int, reduction_uid: int):
    assert (len(tok_ids) > 0 and len(tok_ids) <= 4), "If num_tokens is > 4, please generate two separate schedules for each group of 4 tokens."
    
    NUM_PROCESSORS = len(processors)
    if NUM_PROCESSORS == 1:
        steps = (seq_length + 31) // 32
        duration = PARTIAL_STARTUP_TIME + (steps * PARTIAL_COST_PER_STEP) + PARTIAL_WRITEOUT_TIME
        return [Task(
            uid=partial_uid,
            batch_id=batch_id,
            tok_ids=tok_ids,
            name=f"Partial_B={batch_id}_Tok={tok_ids}_ALL",
            task_type="partial",
            dependencies=[],
            processor=processors[0],
            start=-duration,
            finish=0,
            args={"start": 0, "end": seq_length, "length": seq_length, "write_scratch": False}
        )], partial_uid+1, reduction_uid

    p_idx = 0 # active processor index

    tasks = {p: [] for p in processors} # Note that tasks will be stored in reversed order.

    # Create initial reduction task.
    tasks[processors[p_idx]].append(Task(
        uid=reduction_uid,
        batch_id=batch_id,
        tok_ids=[tok_ids[0]],
        name=f'reduction_{reduction_uid}',
        task_type="reduction",
        dependencies=[],
        processor=processors[p_idx],
        finish=0,
        args={"write_scratch": False}
    ))
    tasks[processors[p_idx]][0].next_input_time = tasks[processors[p_idx]][0].finish - (REDUCTION_WRITEOUT_TIME + REDUCTION_PRODUCER_LATENCY + SYNCHRONIZATION_COST)
    available_inputs = [(-tasks[processors[p_idx]][0].next_input_time, tasks[processors[p_idx]][0])]
    heapq.heapify(available_inputs)
    reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS

    current_cost = __get_quality__([-x[0] for x in available_inputs], num_processors=NUM_PROCESSORS, num_tokens=len(tok_ids), seq_length=seq_length)

    while True:
        # Let's see what would happen if we added a new reduction task.
        speculative_inputs = available_inputs.copy()
        neg_latest_time, parent_task = heapq.heappop(speculative_inputs)
        new_task = Task( # What would this task actually look like?
            uid=reduction_uid,
            batch_id=batch_id,
            tok_ids=[tok_ids[0]],
            name=f'reduction_{reduction_uid}',
            task_type="reduction",
            dependencies=[],
            processor=processors[p_idx],
            finish=-neg_latest_time,
            args={"write_scratch": True}
        )
        new_task.next_input_time = new_task.finish - (REDUCTION_WRITEOUT_TIME + REDUCTION_PRODUCER_LATENCY + SYNCHRONIZATION_COST)
        heapq.heappush(speculative_inputs, (-new_task.next_input_time, new_task)) # None because it's speculative, anyways.
        heapq.heappush(speculative_inputs, (-((-neg_latest_time) - REDUCTION_COST_PER_STEP), parent_task)) # None because it's speculative, anyways.
        spec_cost = __get_quality__([-x[0] for x in speculative_inputs], num_processors=NUM_PROCESSORS, num_tokens=len(tok_ids), seq_length=seq_length)
        if spec_cost > current_cost:
            available_inputs = speculative_inputs # Commit to this new solution.
            parent_task.dependencies = [new_task.uid] + parent_task.dependencies # Add to the start of the dependencies of the parent task.
            parent_task.next_input_time = parent_task.next_input_time - REDUCTION_COST_PER_STEP
            current_cost = spec_cost
            # Actually add the task. But in the meantime, we'll just pretend it's been done.
            tasks[processors[p_idx]].append(new_task)
            reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS # advance indices, now that we are committing.
        else:
            break

    # Next, we need to clone this schedule for all of the other tokens.
    base_reduction_tasks = sorted([t for tasks in tasks.values() for t in tasks], key=lambda x: x.next_input_time, reverse=True)
    task_groups = {t.uid: [t] for t in base_reduction_tasks}
    for i in range(1, len(tok_ids)):
        for task in sorted(base_reduction_tasks, key=lambda x: x.uid):
            new_task = Task(
                uid=reduction_uid,
                batch_id=batch_id,
                tok_ids=[tok_ids[i]],
                name=f'reduction_{reduction_uid}',
                task_type="reduction",
                finish=task.finish,
                next_input_time=task.next_input_time,
                dependencies=[d + i*len(base_reduction_tasks) for d in task.dependencies],
                processor=processors[p_idx],
                args=task.args
            )
            tasks[processors[p_idx]].append(new_task)
            reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS
            task_groups[task.uid].append(new_task)


    # Next we're going to do the round robin, like in get schedule, to assign tasks to processors.
    partial_info = {p: {} for p in processors} # will contain num steps, and the uid of the partial task.
    full_passes, remainder = divmod(NUM_PROCESSORS, len(base_reduction_tasks)) # How many total we'll run through?
    for i in range(full_passes):
        for j in range(len(base_reduction_tasks)):
            for task in task_groups[base_reduction_tasks[j].uid]:
                task.next_input_time -= REDUCTION_COST_PER_STEP
                task.dependencies = [partial_uid] + task.dependencies
            partial_info[processors[p_idx]] = [base_reduction_tasks[j].next_input_time, partial_uid, 0]
            partial_uid, p_idx = partial_uid+1, (p_idx+1) % NUM_PROCESSORS
    for j in range(remainder):
        for task in task_groups[base_reduction_tasks[j].uid]:
            task.next_input_time -= REDUCTION_COST_PER_STEP
            task.dependencies = [partial_uid] + task.dependencies
        partial_info[processors[p_idx]] = [base_reduction_tasks[j].next_input_time, partial_uid, 0]
        partial_uid, p_idx = partial_uid+1, (p_idx+1) % NUM_PROCESSORS
    # at this point, we've assigned all uid's we'll need.

    # Now go backwards through these partials, and assign their end time to be the start time of the reduction
    for k in range(len(tok_ids)):
        for j in reversed(range(len(base_reduction_tasks))):
            p_idx = (p_idx+NUM_PROCESSORS-1) % NUM_PROCESSORS
            actual_start_time = base_reduction_tasks[j].next_input_time + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME # When does this instruction actually start?
            partial_info[processors[p_idx]] = [actual_start_time, partial_info[processors[p_idx]][1], 0]

    # Finally we can go through and assign work to each partial op.
    num_partial_steps = (seq_length + 31) // 32
    # So long as the gap between the earliest and latest partial time is greater than the cost of a step, we should allocate work to the latest one (moving backwards).
    min_val = min([x[0] for x in partial_info.values()]) # Most negative time.
    for k, v in partial_info.items():
        v[2] = min(math.floor((v[0] - min_val) / PARTIAL_COST_PER_STEP), num_partial_steps)
        v[0] -= (v[2] * PARTIAL_COST_PER_STEP)
        num_partial_steps -= v[2]
    
    if num_partial_steps > 0:
        full_passes, remainder = divmod(num_partial_steps, NUM_PROCESSORS)
        for i, p in enumerate(sorted(processors, key=lambda x: partial_info[x][0], reverse=True)):
            num_passes = full_passes + (1 if i<remainder else 0)
            partial_info[p][0] -= (num_passes * PARTIAL_COST_PER_STEP)
            partial_info[p][2] += num_passes
            
    for p in processors:
        for task in tasks[p]:
            task.start = task.next_input_time + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME
            
    # Alright, now that the work has been allocated, we can go through and actually create the tasks.
    current_pos = 0
    for p, v in partial_info.items():
        tasks[p].append(Task(
            uid=v[1],
            batch_id=batch_id,
            tok_ids=tok_ids, # all of them
            name=f'Partial_{v[1]}',
            task_type="partial",
            dependencies=[],
            start=v[0]-PARTIAL_OVERHEAD,
            finish=v[0]+v[2]*PARTIAL_COST_PER_STEP,
            processor=p,
            args={"start": current_pos, "end": min(current_pos+v[2]*32, seq_length), "length": seq_length, "write_scratch": True}
        ))
        current_pos += v[2]*32

    for p, p_tasks in tasks.items():
        earliest_start = p_tasks[-1].start
        num_reductions = 0
        for i, t in enumerate(p_tasks):
            t.start -= earliest_start
            t.finish -= earliest_start
            if t.task_type == "reduction":
                if num_reductions > 0:
                    t.start += p_tasks[i-1].finish - p_tasks[i-1].start
                    t.finish += p_tasks[i-1].finish - p_tasks[i-1].start
                num_reductions += 1

    schedule = [t for tasks in tasks.values() for t in tasks]
    return schedule, partial_uid, reduction_uid


if __name__ == "__main__":
    backward_schedule(list(range(4)), 0, 1024, [0, 1, 2, 3])
    # backward_schedule(list(range(112)), 0, 45096, [0, 1, 2, 3])