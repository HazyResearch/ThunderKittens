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
    RMS_LM_Head,
)
from kvm_runner.llama import LlamaForCausalLM
from kvm_runner.utils import get_sm_count

INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128
NUM_OPS = 6


def pick_num_attention_partitions(prompt_len: int, ntok: int, device: torch.device):
    min_chunk_size = 256
    full_len = prompt_len + ntok

    sm_count = get_sm_count(device)

    num_divisions = math.ceil(full_len / min_chunk_size)

    # TODO limitation until we have a better reduction tree
    num_attention_partitions = min(num_divisions, 24)
    # num_attention_partitions = min(sm_count, num_divisions)

    assert num_attention_partitions >= 1

    return num_attention_partitions


def make_globals(
    model: LlamaForCausalLM,
    skip_attn_reduction: bool = True,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    max_attn_partitions = get_sm_count(device)

    return Globals(
        # model params
        qkv_proj=stacked_params.qkv_proj,
        o_proj=stacked_params.o_proj,
        attn_ln_weight=stacked_params.attn_ln_weight,
        mlp_ln_weight=stacked_params.mlp_ln_weight,
        up_proj=stacked_params.up_proj,
        gate_proj=stacked_params.gate_proj,
        down_proj=stacked_params.down_proj,
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # activation buffers
        hidden_states=make_buffer(config.hidden_size),
        post_ln_rope_q=make_buffer(config.hidden_size),
        attn_out=make_buffer(config.hidden_size),
        attn_out_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions, config.head_dim],
            buffer_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions],
            buffer_dtype=torch.float32,
        ),
        silu_out=make_buffer(config.intermediate_size),
        logits=make_buffer(config.vocab_size),
        # scalars
        pos_id=0,
        attn_scale=1 / math.sqrt(config.head_dim),
        rms_norm_eps=config.rms_norm_eps,
        skip_attn_reduction=skip_attn_reduction,
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
        lm_head_block_size=16,
        matvec_reduction_size=2048,
        attn_kv_block_size=16,
        attn_reduction_size=4,
        vocab_size=config.vocab_size,
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
    if globals.skip_attn_reduction:
        num_attention_partitions = 1
    else:
        num_attention_partitions = pick_num_attention_partitions(
            prompt_len, ntok, globals.device
        )

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

    instruction_1 = True
    instruction_2 = True
    instruction_3 = True
    instruction_4 = True
    instruction_5 = True
    instruction_6 = True

    # INSTRUCTION 1
    if instruction_1:
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

    # INSTRUCTION 2
    if instruction_2:
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

    # INSTRUCTION 3
    # TODO: harcoding one reduction stage, not seqlen dependent
    if instruction_3 and not globals.skip_attn_reduction:
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

    # INSTRUCTION 4
    if instruction_4:
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

    # INSTRUCTION 5
    if instruction_5:
        # print("We love upgate!")
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

    # INSTRUCTION 6
    if instruction_6:
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
    prompt_len: int,
    ntok: int,
    print_info: PrintInfo | None = None,
    globs: Globals | None = None,
    model: LlamaForCausalLM | None = None,
    layer_limit: int | None = None,
    stop_after_op: str | None = None,
):
    if globs is None:
        globals = make_globals(model)
    else:
        globals = globs

    instructions = []

    if layer_limit is None:
        nlayers = globals.num_hidden_layers
    else:
        nlayers = layer_limit

    for layer_idx in range(nlayers):
        instructions.extend(
            schedule_layer(
                globals=globals,
                layer_idx=layer_idx,
                prompt_len=prompt_len,
                ntok=ntok,
                print_info=print_info,
                stop_after_op=stop_after_op,
            )
        )

    if nlayers == globals.num_hidden_layers:
        num_logit_blocks = assert_div(globals.vocab_size, globals.lm_head_block_size)
        for logit_block_idx in range(num_logit_blocks):
            instructions.append(RMS_LM_Head(output_block_idx=logit_block_idx))

    return globals, instructions


def serialize_and_pad(instruction: Instruction):
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instructions: list[Instruction], barrier_init_val: int = 0
):
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

    barriers = (
        torch.ones(
            [
                globs.num_hidden_layers,
                NUM_OPS,
                globs.num_attention_heads + globs.num_kv_heads * 2,
            ],
            dtype=torch.int32,
            device=device,
        )
        * barrier_init_val
    )

    globs.instructions = serialized
    globs.timings = timings
    globs.barriers = barriers
