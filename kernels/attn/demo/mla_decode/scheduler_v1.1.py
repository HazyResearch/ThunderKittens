import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import torch
import math
from scheduler import priority_schedule_tasks as schedule_tasks_forward

# Timing constants (in microseconds)
PARTIAL_STARTUP_TIME = 3.0       # Startup time for partial operations
PARTIAL_WRITEOUT_TIME = 4.5      # Writeout time for partial operations
PARTIAL_COST_PER_STEP = 1.35     # Cost per step (per 32 tokens) for partial operations

REDUCTION_STARTUP_TIME = 2.0     # Startup time for reduction operations
REDUCTION_WRITEOUT_TIME = 1.0    # Writeout time for reduction operations
REDUCTION_COST_PER_STEP = 0.4    # Cost per reduction step

SYNCHRONIZATION_COST = 1.0       # Synchronization cost between dependent operations

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
    # Fields for backward scheduling
    latest_start: float = None
    latest_finish: float = None
    target_finish_time: float = None  # When this task must finish by

def calculate_partial_task_duration(start_pos: int, end_pos: int) -> float:
    """Calculate the duration of a partial task."""
    chunk_length = end_pos - start_pos
    steps = (chunk_length + 31) // 32
    return PARTIAL_STARTUP_TIME + (steps * PARTIAL_COST_PER_STEP) + PARTIAL_WRITEOUT_TIME

def calculate_reduction_task_duration(num_inputs: int) -> float:
    """Calculate the duration of a reduction task."""
    return REDUCTION_STARTUP_TIME + (num_inputs - 1) * REDUCTION_COST_PER_STEP + REDUCTION_WRITEOUT_TIME

def generate_backward_tasks(
    batch_id: int, 
    sequence_length: int, 
    num_processors: int,
    final_time: float = None,
    starting_id: int = 0,
    new_tokens: int = 1
) -> Tuple[List[Task], int]:
    """
    Generates tasks by working backward from the final reduction task.
    
    Args:
        batch_id: Sequence identifier
        sequence_length: Length of the sequence to process
        num_processors: Number of available processors
        final_time: Target finish time for the final task (if None, will be calculated)
        starting_id: First task ID to use
        new_tokens: Number of new tokens to generate
        
    Returns:
        List of tasks and the next available task ID
    """
    tasks = []
    next_id = starting_id
    
    # Keep track of which sequence positions are covered by which partial tasks
    partial_tasks_by_token = [[] for _ in range(new_tokens)]
    
    # Calculate when the final reduction task must finish
    if final_time is None:
        # Default to a reasonable estimate based on sequence length
        final_time = sequence_length * 0.3  # Just an initial guess
    
    # Create the final reduction tasks (one per token)
    final_tasks = []
    for token_id in range(new_tokens):
        final_task = Task(
            uid=next_id,
            batch_id=batch_id,
            name=f"Seq{batch_id}_FinalReduction_tok{token_id}",
            task_type="reduction",
            duration=0,  # Will be updated when we know the number of inputs
            dependencies=[],
            tok_ids=[token_id],
            target_finish_time=final_time,
            args={"write_scratch": False}
        )
        tasks.append(final_task)
        final_tasks.append(final_task)
        next_id += 1
    
    # For each token, recursively generate the task graph backward
    for token_id, final_task in enumerate(final_tasks):
        next_id = generate_reduction_tree_backward(
            final_task,
            tasks,
            batch_id,
            token_id,
            sequence_length,
            num_processors,
            next_id,
            partial_tasks_by_token[token_id]
        )
    
    # Set start and finish times for scheduling
    for task in tasks:
        if task.task_type == "partial":
            # For partial tasks, set directly from duration
            task.finish = task.target_finish_time
            task.start = task.finish - task.duration
        else:
            # For reduction tasks, account for dependencies
            task.finish = task.target_finish_time
            task.start = task.finish - task.duration
    
    # Use a heap-based scheduler to assign processors
    schedule_tasks_forward(tasks, num_processors)
    
    return tasks, next_id

def generate_reduction_tree_backward(
    parent_task: Task,
    all_tasks: List[Task],
    batch_id: int,
    token_id: int,
    sequence_length: int,
    num_processors: int,
    next_id: int,
    partial_tasks: List[Task]
) -> int:
    """
    Recursively generate the reduction tree working backward from a given task.
    
    Args:
        parent_task: The task we're generating inputs for
        all_tasks: List of all tasks (will be modified)
        batch_id: Sequence identifier
        token_id: Token being processed
        sequence_length: Length of sequence
        num_processors: Number of available processors
        next_id: Next available task ID
        partial_tasks: List to collect partial tasks
    
    Returns:
        Next available task ID
    """
    # Calculate how much time we have to generate inputs for this task
    target_time = parent_task.target_finish_time
    
    # Determine optimal number of inputs based on timing and number of processors
    # This is a key optimization point
    
    # Function to evaluate the cost of a given branching factor
    def evaluate_branching_factor(branching: int, depth: int, processors_available: int) -> float:
        """Calculate the cost (time) for a given branching factor."""
        if branching <= 1:
            return float('inf')
        
        # Time required for this reduction
        reduction_time = calculate_reduction_task_duration(branching)
        
        # If we need more reductions, estimate the recursive cost
        if depth > 1:
            # Calculate effective processors after this level
            effective_procs = processors_available - branching
            if effective_procs <= 0:
                return float('inf')
                
            # Cost of next level reduction plus synchronization
            next_branching = min(branching, effective_procs)
            next_level_cost = evaluate_branching_factor(next_branching, depth-1, effective_procs)
            if next_level_cost == float('inf'):
                return float('inf')
                
            return reduction_time + SYNCHRONIZATION_COST + next_level_cost
        else:
            # Final level just needs to do partials
            return reduction_time
    
    # Estimate the depth needed based on sequence length and processors
    # For a balanced tree with p processors, log_b(n) depth is needed
    # where b is the branching factor and n is the sequence length
    num_chunks = max(1, min(num_processors, sequence_length // 32))
    required_depth = max(1, math.ceil(math.log(num_chunks, 2)))
    
    # Try different branching factors to find the most efficient one
    best_branching = 2
    best_cost = float('inf')
    
    for branching in range(2, min(33, num_processors + 1)):
        cost = evaluate_branching_factor(branching, required_depth, num_processors)
        if cost < best_cost:
            best_cost = cost
            best_branching = branching
    
    # For the final reduction task, determine inputs based on covered positions
    if not parent_task.dependencies:
        # If we're at the final task, determine if we need intermediate reductions
        if sequence_length <= 512:  # Small enough to handle with a single reduction
            # Generate partial tasks to cover the sequence and connect directly
            next_id = generate_partial_tasks_backward(
                parent_task, 
                all_tasks,
                batch_id,
                token_id,
                sequence_length,
                next_id,
                partial_tasks
            )
        else:
            # Need intermediate reductions - create a "tree level" of reductions
            target_finish_time = target_time - SYNCHRONIZATION_COST
            branching_factor = best_branching
            
            # Create children reduction tasks
            for i in range(branching_factor):
                segment_start = i * (sequence_length // branching_factor)
                segment_end = (i+1) * (sequence_length // branching_factor) if i < branching_factor-1 else sequence_length
                
                child_task = Task(
                    uid=next_id,
                    batch_id=batch_id,
                    name=f"Seq{batch_id}_Reduction_L1_P{i}_tok{token_id}",
                    task_type="reduction",
                    duration=calculate_reduction_task_duration(2),  # Will be updated later
                    dependencies=[],
                    tok_ids=[token_id],
                    target_finish_time=target_finish_time,
                    args={"write_scratch": True, 
                          "segment_start": segment_start,
                          "segment_end": segment_end}
                )
                all_tasks.append(child_task)
                parent_task.dependencies.append(child_task.uid)
                next_id += 1
                
                # Recursively generate inputs for this reduction
                next_id = generate_reduction_tree_backward(
                    child_task,
                    all_tasks,
                    batch_id,
                    token_id,
                    segment_end - segment_start,
                    num_processors // branching_factor,
                    next_id,
                    partial_tasks
                )
            
            # Update the parent's duration based on actual number of inputs
            parent_task.duration = calculate_reduction_task_duration(len(parent_task.dependencies))
    else:
        # This is an intermediate reduction task
        segment_start = parent_task.args.get("segment_start", 0)
        segment_end = parent_task.args.get("segment_end", sequence_length)
        segment_length = segment_end - segment_start
        
        # Determine if we need further reductions or can use partials
        if segment_length <= 256:  # Small enough to handle with partials
            next_id = generate_partial_tasks_backward(
                parent_task, 
                all_tasks,
                batch_id,
                token_id,
                segment_length,
                next_id,
                partial_tasks,
                segment_start
            )
        else:
            # Need further reduction levels
            target_finish_time = parent_task.target_finish_time - SYNCHRONIZATION_COST
            branching_factor = best_branching
            
            # Create children reduction tasks
            for i in range(branching_factor):
                sub_start = segment_start + i * (segment_length // branching_factor)
                sub_end = segment_start + ((i+1) * (segment_length // branching_factor) if i < branching_factor-1 else segment_length)
                
                child_task = Task(
                    uid=next_id,
                    batch_id=batch_id,
                    name=f"Seq{batch_id}_Reduction_L{parent_task.name.split('_')[2][1:]}_P{i}_tok{token_id}",
                    task_type="reduction",
                    duration=calculate_reduction_task_duration(2),  # Will be updated later
                    dependencies=[],
                    tok_ids=[token_id],
                    target_finish_time=target_finish_time,
                    args={"write_scratch": True,
                          "segment_start": sub_start,
                          "segment_end": sub_end}
                )
                all_tasks.append(child_task)
                parent_task.dependencies.append(child_task.uid)
                next_id += 1
                
                # Recursively generate inputs for this reduction
                next_id = generate_reduction_tree_backward(
                    child_task,
                    all_tasks,
                    batch_id,
                    token_id,
                    sub_end - sub_start,
                    max(1, num_processors // (branching_factor * 2)),
                    next_id,
                    partial_tasks
                )
            
            # Update the parent's duration based on actual number of inputs
            parent_task.duration = calculate_reduction_task_duration(len(parent_task.dependencies))
    
    return next_id

def generate_partial_tasks_backward(
    parent_task: Task,
    all_tasks: List[Task],
    batch_id: int,
    token_id: int,
    segment_length: int,
    next_id: int,
    partial_tasks: List[Task],
    segment_offset: int = 0
) -> int:
    """
    Generate partial tasks to cover a segment, working backward from a reduction task.
    
    Args:
        parent_task: The reduction task these partials will feed into
        all_tasks: List of all tasks (will be modified)
        batch_id: Sequence identifier
        token_id: Token being processed
        segment_length: Length of segment to cover
        next_id: Next available task ID
        partial_tasks: List to collect partial tasks
        segment_offset: Starting position in the overall sequence
        
    Returns:
        Next available task ID
    """
    # Calculate how much time is available for partial tasks
    target_time = parent_task.target_finish_time - SYNCHRONIZATION_COST
    
    # Determine optimal chunk size for partials
    # Balance between having enough chunks for parallelism
    # and minimizing overhead from too many small chunks
    
    # Consider available processors and timing constants
    optimal_partial_size = max(32, min(256, segment_length // 16))
    
    # Create chunking strategy
    chunks = []
    for start in range(0, segment_length, optimal_partial_size):
        end = min(start + optimal_partial_size, segment_length)
        chunks.append((start, end))
    
    # Generate partial tasks for each chunk
    for chunk_start, chunk_end in chunks:
        abs_start = segment_offset + chunk_start
        abs_end = segment_offset + chunk_end
        chunk_length = chunk_end - chunk_start
        
        # Calculate duration for this partial task
        duration = calculate_partial_task_duration(chunk_start, chunk_end)
        
        # Calculate target finish time for this partial
        # All partials must finish by the parent's target time minus synchronization
        partial_task = Task(
            uid=next_id,
            batch_id=batch_id,
            name=f"Seq{batch_id}_Partial_{abs_start}-{abs_end}_tok{token_id}",
            task_type="partial",
            duration=duration,
            dependencies=[],
            tok_ids=[token_id],
            target_finish_time=target_time,
            args={"start": abs_start,
                  "end": abs_end,
                  "write_scratch": True,
                  "length": segment_length}
        )
        
        all_tasks.append(partial_task)
        parent_task.dependencies.append(partial_task.uid)
        partial_tasks.append(partial_task)
        next_id += 1
    
    # Update the parent task's duration based on actual number of inputs
    parent_task.duration = calculate_reduction_task_duration(len(parent_task.dependencies))
    
    return next_id

# def schedule_tasks_forward(tasks: List[Task], num_processors: int, next_id: int) -> int:
#     """
#     Schedule tasks using forward scheduling based on dependencies.
#     Takes into account the target finish times from the backward pass.
    
#     Args:
#         tasks: List of all tasks
#         num_processors: Number of available processors
#         next_id: Next available task ID
        
#     Returns:
#         Next available task ID
#     """
#     tasks_by_id = {t.uid: t for t in tasks}
    
#     # Build dependency graph for topological sort
#     in_degree = {t.uid: 0 for t in tasks}
#     dependents = {t.uid: [] for t in tasks}
    
#     for task in tasks:
#         for dep_id in task.dependencies:
#             in_degree[task.uid] += 1
#             dependents[dep_id].append(task.uid)
    
#     # Find tasks with no dependencies
#     ready = [t for t in tasks if in_degree[t.uid] == 0]
    
#     # Initialize processor availability
#     processors = [(0, pid) for pid in range(num_processors)]
#     heapq.heapify(processors)
    
#     # Maintain critical path information
#     critical_path = set()
    
#     # Find tasks on the critical path (those with largest target_finish_time)
#     if tasks:
#         max_target_time = max(t.target_finish_time or 0 for t in tasks)
#         critical_path = {t.uid for t in tasks if t.target_finish_time == max_target_time}
        
#         # Add dependencies of critical tasks to the critical path
#         for task in list(tasks):
#             if task.uid in critical_path:
#                 for dep_id in task.dependencies:
#                     critical_path.add(dep_id)
    
#     # Schedule tasks
#     while ready:
#         # Prioritize tasks on critical path first, then by dependencies
#         ready.sort(key=lambda t: (
#             0 if t.uid in critical_path else 1,  # Critical path tasks first
#             -sum(1 for dep_id in dependents[t.uid] if in_degree[dep_id] == 1),  # Tasks that unblock others
#             -t.target_finish_time if t.target_finish_time else 0  # Earliest deadline
#         ))
        
#         task = ready.pop(0)
        
#         # Get earliest available processor
#         current_time, proc_id = heapq.heappop(processors)
        
#         # Calculate earliest start time based on dependencies
#         earliest_deps_done = max([0] + [tasks_by_id[dep_id].finish + SYNCHRONIZATION_COST 
#                                        for dep_id in task.dependencies])
#         start_time = max(current_time, earliest_deps_done)
        
#         # Calculate finish time
#         finish_time = start_time + task.duration
        
#         # Check if we can meet the target finish time
#         if task.target_finish_time and finish_time > task.target_finish_time:
#             # We'll miss the deadline - try to optimize
#             # This might involve rearranging tasks or using different processors
#             # For simplicity, just log a warning for now
#             print(f"Warning: Task {task.name} will finish at {finish_time:.2f}, " 
#                   f"missing target of {task.target_finish_time:.2f}")
        
#         # Update task timing
#         task.start = start_time
#         task.finish = finish_time
#         task.processor = proc_id
        
#         # Update processor availability
#         heapq.heappush(processors, (finish_time, proc_id))
        
#         # Update dependent tasks
#         for dep_id in dependents[task.uid]:
#             in_degree[dep_id] -= 1
#             if in_degree[dep_id] == 0:
#                 ready.append(tasks_by_id[dep_id])
    
#     return next_id