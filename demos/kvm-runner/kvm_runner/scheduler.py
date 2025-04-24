import math

import torch
from kvm_runner.instructions import (
    AttentionReduction,
    DownProjResidual,
    Globals,
    Instruction,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    PrintInfo,
    PrintState,
)
from kvm_runner.llama import LlamaForCausalLM
from kvm_runner.utils import get_sm_count

INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128
NUM_OPS = 6


def make_globals(
    model: LlamaForCausalLM,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    max_attn_partials = get_sm_count(device)

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    return Globals(
        # model params
        qkv_proj=stacked_params.qkv_proj,
        o_proj=stacked_params.o_proj,
        attn_ln_weight=stacked_params.attn_ln_weight,
        mlp_ln_weight=stacked_params.mlp_ln_weight,
        up_proj=stacked_params.up_proj,
        gate_proj=stacked_params.gate_proj,
        down_proj=stacked_params.down_proj,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # activation buffers
        hidden_states=make_buffer(config.hidden_size),
        post_ln_rope_q=make_buffer(config.hidden_size),
        attn_out=make_buffer(config.hidden_size),
        attn_out_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partials, config.head_dim],
            buffer_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partials], buffer_dtype=torch.float32
        ),
        silu_out=make_buffer(config.intermediate_size),
        # scalars
        pos_id=0,
        attn_scale=1 / math.sqrt(config.head_dim),
        rms_norm_eps=config.rms_norm_eps,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        # block sizes
        up_gate_proj_block_size=16,
        down_proj_block_size=16,
        qkv_block_size=16,
        o_proj_block_size=16,
        matvec_reduction_size=2048,
        attn_kv_block_size=16,
        attn_reduction_size=4,
        device=device,
    )


def assert_div(a, b):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def schedule_layer(
    globals: Globals,
    layer_idx: int,
    prompt_len: int,
    ntok: int,
    stop_after_op: str | None = None,
    print_info: PrintInfo | None = None,
):
    num_attention_partitions = 2
    add_print_instructions = print_info is not None

    instructions: list[Instruction] = []

    qkv_outdim = (
        globals.num_attention_heads + 2 * globals.num_kv_heads
    ) * globals.head_dim

    def maybe_add_print(layer_idx: int, name: str):
        if add_print_instructions:
            instructions.append(
                PrintState(
                    layer_idx=layer_idx,
                    name=name,
                    print_info=print_info,
                )
            )

    num_qkv_blocks = assert_div(qkv_outdim, globals.qkv_block_size)
    for qkv_block_idx in range(num_qkv_blocks):
        instructions.append(
            LayerNorm_QKV_MatVecRopeAppend(
                layer_idx=layer_idx,
                output_block_idx=qkv_block_idx,
            )
        )

    maybe_add_print(layer_idx, "qkv")

    if stop_after_op == "qkv":
        return instructions

    for kv_head_idx in range(globals.num_kv_heads):
        for partial_idx in range(num_attention_partitions):
            instructions.append(
                PartialAttention(
                    layer_idx=layer_idx,
                    kv_head_idx=kv_head_idx,
                    num_partials=num_attention_partitions,
                    partial_idx=partial_idx,
                )
            )

    maybe_add_print(layer_idx, "partial_attn")

    if stop_after_op == "partial_attn":
        return instructions

    # HACK: one reduction stage, for correctness
    for head_start_idx in range(
        0, globals.num_attention_heads, globals.attn_reduction_size
    ):
        instructions.append(
            AttentionReduction(
                layer_idx=layer_idx,
                head_start_idx=head_start_idx,
                is_terminal=True,
                num_partials=num_attention_partitions,
                reduction_list=list(range(num_attention_partitions)),
            ),
        )

    maybe_add_print(layer_idx, "attn_reduction")

    if stop_after_op == "attn_reduction":
        return instructions

    num_o_blocks = assert_div(globals.hidden_size, globals.o_proj_block_size)
    for o_block_idx in range(num_o_blocks):
        instructions.append(
            O_ProjResidual(
                layer_idx=layer_idx,
                output_block_idx=o_block_idx,
                reduction_block_idx=0,
            )
        )

    maybe_add_print(layer_idx, "o_proj")

    if stop_after_op == "o_proj":
        return instructions

    num_up_gate_blocks = assert_div(
        globals.intermediate_size, globals.up_gate_proj_block_size
    )
    for up_gate_block_idx in range(num_up_gate_blocks):
        instructions.append(
            LayerNormDoubleMatVecSiLU(
                layer_idx=layer_idx,
                output_block_idx=up_gate_block_idx,
            )
        )

    maybe_add_print(layer_idx, "up_gate")

    if stop_after_op == "up_gate":
        return instructions

    num_down_blocks = assert_div(globals.hidden_size, globals.down_proj_block_size)
    for down_block_idx in range(num_down_blocks):
        for reduction_idx in range(4):  # 2048 columns per op
            instructions.append(
                DownProjResidual(
                    layer_idx=layer_idx,
                    output_block_idx=down_block_idx,
                    reduction_block_idx=reduction_idx,
                )
            )

    maybe_add_print(layer_idx, "down_proj")

    if stop_after_op == "down_proj":
        return instructions

    return instructions


def schedule_model(
    model: LlamaForCausalLM,
    prompt_len: int,
    ntok: int,
    print_info: PrintInfo | None = None,
):
    config = model.config
    globals = make_globals(model)
    instructions = []

    for layer_idx in range(config.num_hidden_layers):
        instructions.extend(
            schedule_layer(
                globals=globals,
                layer_idx=layer_idx,
                prompt_len=prompt_len,
                ntok=ntok,
                print_info=print_info,
            )
        )

    return globals, instructions


def serialize_and_pad(instruction: Instruction):
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(globs: Globals, instructions: list[Instruction]):
    num_sms = get_sm_count(globs.device)
    instruction_queues = [[] for _ in range(num_sms)]
    for i, instruction in enumerate(instructions):
        instruction_queues[i % num_sms].append(serialize_and_pad(instruction))

    empty_instruction = [0] * INTS_PER_INSTRUCTION

    max_queue_len = max(len(queue) for queue in instruction_queues)
    for queue in instruction_queues:
        queue.extend([empty_instruction] * (max_queue_len - len(queue)))

    flattened = []
    for queue in instruction_queues:
        for instruction in queue:
            flattened.extend(instruction)

    device = globs.device

    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )

    timings = torch.zeros(
        [num_sms, max_queue_len, TIMING_SLOTS],
        dtype=torch.int32,
        device=device,
    )

    barriers = torch.zeros(
        [
            globs.num_hidden_layers,
            NUM_OPS,
            globs.num_attention_heads + globs.num_kv_heads * 2,
        ],
        dtype=torch.int32,
        device=device,
    )

    globs.instructions = serialized
    globs.timings = timings
    globs.barriers = barriers
