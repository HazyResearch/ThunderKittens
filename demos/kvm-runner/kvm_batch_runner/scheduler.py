import math
import torch
from kvm_batch_runner.instructions import (
    PreLayerNorm,
    QKV_MatMulRopeAppend,
    AttentionDecode,
    O_ProjResidual,

    PreMLPLayerNorm,
    GateSilu,
    UpMatMul,
    DownProjResidual,

    Globals,
    Instruction,
    PrintInfo,
    PrintState,
    RMS_LM_Head,
)
from kvm_batch_runner.llama import LlamaForCausalLM
from kvm_batch_runner.utils import get_sm_count

INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128
NUM_OPS = 8


def make_globals(
    model: LlamaForCausalLM,
    extra_config,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    max_attn_partials = get_sm_count(device)

    def make_buffer(shape_bsz, shape, buffer_dtype=dtype):
        return torch.zeros((shape_bsz, shape), device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    '''
    Need to keep track of things for paged KV cache... 



    '''

    MAX_BLOCKS_PER_SEQUENCE = 1024

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
        hidden_states=make_buffer(extra_config.max_batch_size, config.hidden_size),
        post_ln_rope_q=make_buffer(extra_config.max_batch_size, config.hidden_size),
        attn_out=make_buffer(extra_config.max_batch_size, config.hidden_size),
        silu_out=make_buffer(extra_config.max_batch_size, config.intermediate_size),
        logits=make_buffer(extra_config.max_batch_size, config.vocab_size),

        # paged kv cache
        page_table=make_buffer(extra_config.max_batch_size, MAX_BLOCKS_PER_SEQUENCE),
        next_free_page = torch.zeros(1, dtype=torch.uint32, device=device)

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
        matmul_silu_block_size=128,
        matmul_gate_block_size=128,
        down_proj_block_size=128,
        qkv_block_size=128,
        o_proj_block_size=128,
        lm_head_block_size=128,
        matmul_block_size=128,
        attn_kv_block_size=16,

        batch_block_size=128,

        # attn_reduction_size=4,
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
    batch_size: int = 1,
):
    # min_chunk_size = 16
    full_len = prompt_len + ntok
    sm_count = get_sm_count(globals.device)

    # num_divisions = math.ceil(full_len / min_chunk_size)

    # TODO limitation until we have a better reduction tree
    # num_attention_partitions = min(num_divisions, 24)
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
    instruction_7 = True
    instruction_8 = True

    ###########################################
    # ATTENTION
    ###########################################

    # total_batch_size = 512
    # batch_block_size = 128
    # num_minibatch = 4 


    # TODO: 
    # - do math for batches per op
    # - want longer sequence lengths for more compute in attention

    # INSTRUCTION 1
    if instruction_1:
        for bidx in range(0, batch_size):
            instructions.append(
                PreLayerNorm(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx
                )
            )
            maybe_add_print(layer_idx, "pre_attn_layernorm")
            if stop_after_op == "pre_attn_layernorm":
                return instructions


    # INSTRUCTION 2
    if instruction_2:
        for bidx in range(0, batch_size, globals.batch_block_size):
            num_qkv_blocks = assert_div(qkv_outdim, globals.qkv_block_size)
            for qkv_block_idx in range(num_qkv_blocks):
                instructions.append(
                    QKV_MatMulRopeAppend(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        qkv_block_idx=qkv_block_idx,
                    )
                )
            maybe_add_print(layer_idx, "qkv_rope_append")
            if stop_after_op == "qkv_rope_append":
                return instructions


    # INSTRUCTION 3
    # Need to figure out how to give in kv_indices
    if instruction_3:
        for bidx in range(0, batch_size, globals.batch_block_size):
            for kv_head_idx in range(globals.num_kv_heads):
                instructions.append(
                    AttentionDecode(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        kv_head_idx=kv_head_idx
                    ),
                )
            maybe_add_print(layer_idx, "attn_decode")
            if stop_after_op == "attn_decode":
                return instructions


    # INSTRUCTION 4
    if instruction_4:
        for bidx in range(0, batch_size, globals.batch_block_size):
            num_o_blocks = assert_div(globals.hidden_size, globals.o_proj_block_size)
            for o_block_idx in range(num_o_blocks):
                instructions.append(
                    O_ProjResidual(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        output_block_idx=o_block_idx,
                    )
                )
            maybe_add_print(layer_idx, "o_proj")
            if stop_after_op == "o_proj":
                return instructions


    ###########################################
    # MLP
    ###########################################

    # INSTRUCTION 5
    if instruction_5:
        for bidx in range(0, batch_size):
            instructions.append(
                PreMLPLayerNorm(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                )
            )
            maybe_add_print(layer_idx, "pre_mlp_layernorm")
            if stop_after_op == "pre_mlp_layernorm":
                return instructions


    # INSTRUCTION 6
    if instruction_6:
        for bidx in range(0, batch_size, globals.batch_block_size):
            num_matmul_silu_blocks = assert_div(globals.intermediate_size, globals.matmul_silu_block_size)
            for block_idx in range(num_matmul_silu_blocks):
                instructions.append(
                    GateSilu(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        output_block_idx=block_idx,
                    )
                )
            maybe_add_print(layer_idx, "gate_silu")
            if stop_after_op == "gate_silu":
                return instructions


    # INSTRUCTION 7
    if instruction_7:
        for bidx in range(0, batch_size, globals.batch_block_size):
            num_matmul_gate_blocks = assert_div(globals.intermediate_size, globals.matmul_gate_block_size)
            for block_idx in range(num_matmul_gate_blocks):
                instructions.append(
                    UpMatMul(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        output_block_idx=block_idx,
                    )
                )
            maybe_add_print(layer_idx, "up_matmul")
            if stop_after_op == "up_matmul":
                return instructions



    # INSTRUCTION 8
    if instruction_8:
        for bidx in range(0, batch_size, globals.batch_block_size):
            num_down_blocks = assert_div(globals.hidden_size, globals.down_proj_block_size)
            for down_block_idx in range(num_down_blocks):
                instructions.append(
                    DownProjResidual(
                        layer_idx=layer_idx,
                        batch_start_idx=bidx,
                        output_block_idx=down_block_idx,
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
        globals = make_globals(model, model.extra_config)
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
                batch_size = model.extra_config.max_batch_size,
            )
        )

    if nlayers == globals.num_hidden_layers:
        num_logit_blocks = assert_div(globals.vocab_size, globals.lm_head_block_size)
        for bidx in range(0, model.extra_config.max_batch_size // globals.batch_block_size, globals.batch_block_size):
            for logit_block_idx in range(num_logit_blocks):
                instructions.append(RMS_LM_Head(
                    output_block_idx=logit_block_idx,
                    batch_start_idx=bidx,
                    batch_end_idx=bidx + globals.batch_block_size,
                ))

    return globals, instructions


def serialize_and_pad(instruction: Instruction):
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: Globals, instructions: list[Instruction], 
    batch_size: int, barrier_init_val: int = 0
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
                batch_size // globs.batch_block_size,
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
