import torch
from einops import einsum
from kvm_unity.instructions import BaseGlobals, Instruction, PrintState
from kvm_unity.model_types import BatchState
from kvm_unity.utils import trepr
from torch import Tensor


def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matvec(
    mat: Tensor,
    vec: Tensor,
    block_size: int,
    block_idx: int,
    reduce: bool = False,
    reduction_size: int = 0,
    reduction_idx: int = 0,
):
    start, end = get_start_end(block_size, block_idx)
    if reduce:
        red_start, red_end = get_start_end(reduction_size, reduction_idx)
        mat = mat[start:end, red_start:red_end]
        vec = vec[red_start:red_end]
    else:
        mat = mat[start:end]

    out = einsum(mat, vec, "o i, i -> o")
    return out, start, end


def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)


def matvec_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    block_size: int,
    start_block_idx: int,
    end_block_idx: int,
    reduction_size: int,
    reduction_block_idx: int,
):
    for block_idx in range(start_block_idx, end_block_idx):
        matvec_out, start, end = matvec(
            mat=mat,
            vec=vec,
            block_size=block_size,
            block_idx=block_idx,
            reduce=True,
            reduction_size=reduction_size,
            reduction_idx=reduction_block_idx,
        )

        residual[start:end] += matvec_out.to(residual.dtype)


def print_state(globals: BaseGlobals, instruction: PrintState):
    print_info = instruction.print_info
    if (
        print_info.layer_filter is None
        or instruction.layer_idx in print_info.layer_filter
    ) and (
        print_info.name_filter is None or instruction.name in print_info.name_filter
    ):
        print(f"State at layer={instruction.layer_idx}, op={instruction.name}")
        for state in print_info.state_filter:
            attr = getattr(globals, state)
            print(f"{state}: {trepr(attr) if isinstance(attr, Tensor) else attr}")


def interpret_with_pyvm(
    globals: BaseGlobals, instructions: list[Instruction], instruction_to_solver: dict
):
    for instruction in instructions:
        instruction_to_solver[type(instruction)](globals, instruction)


class PyVM_Interpreter:
    def __init__(self, instruction_to_solver: dict):
        self.instruction_to_solver = instruction_to_solver

    def interpret(self, globs: BaseGlobals, instructions: list[Instruction]):
        interpret_with_pyvm(globs, instructions, self.instruction_to_solver)


class PyVM_Runner:
    def __init__(self, model: LlamaForCausalLM, schedule: Schedule):
        self.model = model

        self.schedule = Schedule(
            globs=make_globals(self.model),
            dag_nodes=make_dag(self.model, prompt_len, ntok),
        )

        queues = assign_to_sms(sched, self.schedule)
        tensorize_instructions(self.schedule.globs, queues)

        self.instructions = self.schedule.get_linear_instructions()

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )

        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.schedule.globs.hidden_states[:] = hiddens
        self.schedule.globs.barriers.zero_()
        self.schedule.globs.pos_id = pos_id

        interpret_with_pyvm(self.schedule.globs, self.instructions)

        output_hiddens = self.schedule.globs.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids
