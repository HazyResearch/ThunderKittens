import heapq
from dataclasses import dataclass, field, replace

import torch

from kvm_unity.instructions import (
    BaseGlobals,
    Instruction,
    NoOp,
)
from kvm_unity.llama import LlamaForCausalLM

INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128


@dataclass
class DAG_Node:
    def __hash__(self):
        return hash(tuple(self.instruction.serialize()))

    instruction: Instruction
    dependencies: list["DAG_Node"]

    children: set["DAG_Node"] = field(default_factory=set)
    start_time: float = float("inf")
    end_time: float = float("inf")
    remaining_dependencies: set["DAG_Node"] = field(default_factory=set)
    priority: float = 0

    def earliest_ready_time(self, globs: BaseGlobals):
        if len(self.dependencies) == 0:
            return 0

        return max(dep.end_time for dep in self.dependencies)

    def register_with_parents(self):
        for dep in self.dependencies:
            dep.children.add(self)

    def calc_priority(self, globs: BaseGlobals):
        cur_cost = self.priority
        for dep in self.dependencies:
            pri = cur_cost + dep.instruction.cost(globs)
            dep.priority = max(pri, dep.priority)
            dep.calc_priority(globs)


@dataclass
class Schedule:
    globs: BaseGlobals
    dag_nodes: list[DAG_Node]
    end_node: DAG_Node

    def get_linear_instructions(self):
        # NOTE: assumes this is in topological order
        return [node.instruction for node in self.dag_nodes]

    def smart_assign_to_sms(self):
        return assign_dag_to_sms(self)

    def round_robin_assign_to_sms(self):
        instructions = self.get_linear_instructions()
        return round_robin_assign_to_sms(instructions, self.globs.sm_count())


class ScheduleBuilder:
    @classmethod
    def make_globals(cls, model):
        raise NotImplementedError

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None
    ):
        raise NotImplementedError

    @classmethod
    def build(
        cls,
        model: LlamaForCausalLM,
        stop_after_op: str | None = None,
        layer_limit: int | None = None,
    ):
        globs = cls.make_globals(model)
        dag_nodes, end_node = cls.make_dag(globs, stop_after_op, layer_limit)
        return Schedule(globs, dag_nodes, end_node)

    @classmethod
    def with_new_globals(cls, schedule: Schedule, model: LlamaForCausalLM):
        return replace(schedule, globs=cls.make_globals(model))


def assign_dag_to_sms(schedule: Schedule) -> list[list[Instruction]]:
    nodes = schedule.dag_nodes
    globs = schedule.globs

    for node in nodes:
        node.register_with_parents()

    for node in nodes:
        node.remaining_dependencies = set(node.dependencies)

    # schedule.end_node.calc_priority(globs)

    sm_count = globs.sm_count()
    sm_queues = [[] for _ in range(sm_count)]

    sm_heap = [(0, i) for i in range(sm_count)]
    heapq.heapify(sm_heap)

    ready_nodes = [n for n in nodes if len(n.dependencies) == 0]

    idx_to_node = {i: n for i, n in enumerate(nodes)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ready_heap = []
    for node in ready_nodes:
        idx = node_to_idx[node]
        # max cost first
        ready_heap.append((-node.instruction.cost(globs), idx))

    heapq.heapify(ready_heap)

    while ready_heap:
        ready_time, idx = heapq.heappop(ready_heap)
        node = idx_to_node[idx]

        sm_time, sm_idx = heapq.heappop(sm_heap)

        # print(f"assigning instruction {node.instruction} to sm {sm_idx}")

        # start_time = max(ready_time, sm_time)
        start_time = sm_time

        end_time = start_time + node.instruction.cost(globs)

        node.start_time = start_time
        node.end_time = end_time

        sm_queues[sm_idx].append(node.instruction)

        heapq.heappush(sm_heap, (end_time, sm_idx))

        for child in node.children:
            child.remaining_dependencies.remove(node)
            if len(child.remaining_dependencies) == 0:
                idx = node_to_idx[child]
                heapq.heappush(ready_heap, (-child.instruction.cost(globs), idx))

    return sm_queues


def round_robin_assign_to_sms(
    instructions: list[Instruction], sm_count: int
) -> list[list[Instruction]]:
    sm_queues = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        sm_queues[i % sm_count].append(instruction)

    return sm_queues


def zig_zag_assign_to_sms(
    instructions: list[Instruction], sm_count: int
) -> list[list[Instruction]]:
    sm_queues = [[] for _ in range(sm_count)]
    for i, instruction in enumerate(instructions):
        base_id = i % (sm_count * 2)
        if base_id < sm_count:
            sm_queues[base_id].append(instruction)
        else:
            sm_queues[sm_count - 1 - (base_id - sm_count)].append(instruction)

    return sm_queues


def collect_into_waves(instructions: list[Instruction]):
    waves: list[list[Instruction]] = []
    cur = []
    for instruction in instructions:
        if cur == [] or cur[-1].opcode() == instruction.opcode():
            cur.append(instruction)
        else:
            waves.append(cur)
            cur = [instruction]

    if len(cur) > 0:
        waves.append(cur)

    return waves


def wave_assign_to_sms(
    schedule: Schedule,
) -> list[list[Instruction]]:
    instructions = schedule.get_linear_instructions()
    globs = schedule.globs
    sm_count = globs.sm_count()

    waves = collect_into_waves(instructions)

    sm_queues = [[] for _ in range(sm_count)]

    sm_heap = [(0, i) for i in range(sm_count)]
    heapq.heapify(sm_heap)

    for wave in waves:
        sorted_by_biggest_cost = sorted(wave, key=lambda x: x.cost(globs), reverse=True)

        for ins in sorted_by_biggest_cost:
            sm_cost, sm_idx = heapq.heappop(sm_heap)
            sm_cost += ins.cost(globs)
            heapq.heappush(sm_heap, (sm_cost, sm_idx))
            sm_queues[sm_idx].append(ins)

    return sm_queues


def pool_assign_to_sms(
    instructions: list[Instruction], sm_count: int, memory_fraction: float
) -> list[list[Instruction]]:
    memory_instructions = []
    compute_instructions = []

    for ins in instructions:
        pool = ins.tags()["pool"]
        match pool:
            case "memory":
                memory_instructions.append(ins)
            case "compute":
                compute_instructions.append(ins)
            case _:
                raise ValueError(f"Unknown pool: {pool}")

    mem_sms = round(sm_count * memory_fraction)
    compute_sms = sm_count - mem_sms

    memory_queues = round_robin_assign_to_sms(memory_instructions, mem_sms)
    compute_queues = round_robin_assign_to_sms(compute_instructions, compute_sms)

    return memory_queues + compute_queues


def assign_to_sms(
    mode: str,
    schedule: Schedule | None = None,
    instructions: list[Instruction] | None = None,
    sm_count: int | None = None,
    memory_fraction: float | None = None,
):
    if schedule is not None:
        instructions = schedule.get_linear_instructions()
        sm_count = schedule.globs.sm_count()

    match mode:
        case "rr":
            return round_robin_assign_to_sms(instructions, sm_count)
        case "zz":
            return zig_zag_assign_to_sms(instructions, sm_count)
        case "wave":
            return wave_assign_to_sms(schedule)
        case "dag":
            return assign_dag_to_sms(schedule)
        case "pool":
            assert memory_fraction is not None
            return pool_assign_to_sms(
                instructions, sm_count, memory_fraction=memory_fraction
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def serialize_and_pad(instruction: Instruction):
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: BaseGlobals,
    instruction_queues: list[list[Instruction]],
):
    num_sms = globs.sm_count()

    max_queue_len = max(len(queue) for queue in instruction_queues)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    flattened = []
    for queue in instruction_queues:
        flattened.extend(serialize_and_pad(instruction) for instruction in queue)

    device = globs.device

    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )

    timings = torch.zeros(
        [num_sms, max_queue_len, TIMING_SLOTS],
        dtype=torch.int32,
        device=device,
    )

    globs.instructions = serialized
    globs.timings = timings
