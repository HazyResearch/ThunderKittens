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


def make_globals(
    model: LlamaForCausalLM,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    max_attn_partials = 1024
    max_barriers = 1024 * 128
    max_instructions = 1024 * 128
    max_timings = 1024 * 128

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    return Globals(
        # model params
        # qkv_proj=stacked_sd["model.layers.all.self_attn.qkv_proj.weight"],
        # attn_ln_weight=stacked_sd["model.layers.all.self_attn.norm_attn.weight"],
        # o_proj=stacked_sd["model.layers.all.self_attn.out_proj.weight"],
        # mlp_ln_weight=stacked_sd["model.layers.all.mlp.norm_mlp.weight"],
        # up_proj=stacked_sd["model.layers.all.mlp.up_proj.weight"],
        # gate_proj=stacked_sd["model.layers.all.mlp.gate_proj.weight"],
        # down_proj=stacked_sd["model.layers.all.mlp.down_proj.weight"],
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
            config.num_attention_heads, max_attn_partials, buffer_dtype=torch.float32
        ),
        silu_out=make_buffer(config.intermediate_size),
        # scalars
        pos_id=0,
        softmax_temp=1 / math.sqrt(config.head_dim),
        ln_eps=config.rms_norm_eps,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        # block sizes
        up_gate_proj_block_size=128,
        down_proj_block_size=128,
        qkv_block_size=128,
        o_proj_block_size=128,
        attn_kv_block_size=16,
        # misc buffers
        instructions=make_buffer(max_instructions),
        barriers=make_buffer(max_barriers),
        timings=make_buffer(max_timings),
        # max sizes
        max_attn_partials=max_attn_partials,
        max_barriers=max_barriers,
        max_instructions=max_instructions,
        max_timings=max_timings,
    )


def assert_div(a, b):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def schedule_layers(
    globals: Globals,
    layer_idx: int,
    num_attention_partitions: int,
    print_info: PrintInfo | None = None,
):
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

    maybe_add_print(layer_idx, "attn")

    # HACK: one reduction stage, for correctness
    for head_idx in range(globals.num_attention_heads):
        instructions.append(
            AttentionReduction(
                layer_idx=layer_idx,
                head_idx=head_idx,
                is_terminal=True,
                num_partials=num_attention_partitions,
                reduction_list=list(range(num_attention_partitions)),
            ),
        )

    maybe_add_print(layer_idx, "attn_reduction")

    num_o_blocks = assert_div(globals.hidden_size, globals.o_proj_block_size)
    for o_block_idx in range(num_o_blocks):
        instructions.append(
            O_ProjResidual(
                layer_idx=layer_idx,
                output_block_idx=o_block_idx,
            )
        )

    maybe_add_print(layer_idx, "o_proj")

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

    num_down_blocks = assert_div(globals.hidden_size, globals.down_proj_block_size)
    for down_block_idx in range(num_down_blocks):
        instructions.append(
            DownProjResidual(
                layer_idx=layer_idx,
                output_block_idx=down_block_idx,
            )
        )

    maybe_add_print(layer_idx, "down_proj")

    return instructions


def schedule_model(
    model: LlamaForCausalLM,
    num_attention_partitions: int,
    print_info: PrintInfo | None = None,
):
    config = model.config
    globals = make_globals(model)
    instructions = []

    for layer_idx in range(config.num_hidden_layers):
        instructions.extend(
            schedule_layers(
                globals=globals,
                layer_idx=layer_idx,
                num_attention_partitions=num_attention_partitions,
                print_info=print_info,
            )
        )

    return globals, instructions
