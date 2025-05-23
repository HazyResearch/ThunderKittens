import math

import torch
from kvm_unity.instructions import NoOp
from kvm_unity.llama import LlamaForCausalLM
from kvm_unity.scheduler import DAG_Node, ScheduleBuilder
from kvm_unity.throughput.instructions import (
    AttentionDecode,
    DownProjResidual,
    GateSilu,
    Globals,
    Instruction,
    LM_Head,
    O_ProjResidual,
    PreAttnLayerNorm,
    PreLMHeadRMS,
    PreMLP_Norm,
    QKV_MatMulRopeAppend,
    UpMatMul,
)
from kvm_unity.utils import assert_div


def make_globals(
    model: LlamaForCausalLM,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    def make_buffer(shape_bsz, shape, buffer_dtype=dtype):
        return torch.zeros((shape_bsz, shape), device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    extra_config = model.extra_config
    bs = extra_config.max_batch_size

    matmul_batch_block_size = 128
    matmul_output_block_size = 128
    norm_block_size = 16

    barriers = torch.zeros(
        [
            config.num_hidden_layers,
            10,  # more than the number of opcodes we have
            bs,
            max(
                config.num_attention_heads + config.num_key_value_heads * 2,
                assert_div(config.intermediate_size, matmul_output_block_size),
            ),
        ],
        dtype=torch.int32,
        device=device,
    )

    return Globals(
        # model params
        qkv_proj_weights=stacked_params.qkv_proj,
        o_proj_weights=stacked_params.o_proj,
        attn_ln_weights=stacked_params.attn_ln_weight,
        mlp_ln_weights=stacked_params.mlp_ln_weight,
        up_proj_weights=stacked_params.up_proj,
        gate_proj_weights=stacked_params.gate_proj,
        down_proj_weights=stacked_params.down_proj,
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # activation buffers
        hidden_states=make_buffer(bs, config.hidden_size),
        rms_rope_intermediates=make_buffer(bs, config.hidden_size),
        rms_gate_intermediates=make_buffer(bs, config.hidden_size),
        rms_lm_head_intermediates=make_buffer(bs, config.hidden_size),
        post_ln_rope_q=make_buffer(bs, config.hidden_size),
        attn_out=make_buffer(bs, config.hidden_size),
        silu_out=make_buffer(bs, config.intermediate_size),
        logits=make_buffer(bs, config.vocab_size),
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
        batch_size=bs,
        matmul_batch_block_size=matmul_batch_block_size,
        matmul_output_block_size=matmul_output_block_size,
        norm_block_size=norm_block_size,
        vocab_size=config.vocab_size,
        device=device,
        barriers=barriers,
    )


def schedule_pre_attn_norm(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:  # Returns instructions and a flag to stop
    instructions: list[Instruction] = []

    for bidx in range(0, globs.batch_size):
        instructions.append(PreAttnLayerNorm(layer_idx=layer_idx, batch_start_idx=bidx))

    return instructions


def schedule_qkv_matmul_rope_append(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    num_batch_blocks = assert_div(globs.batch_size, globs.matmul_batch_block_size)
    num_qkv_blocks = assert_div(
        globs.qkv_dim(),
        globs.matmul_output_block_size,
    )

    for bidx in range(0, num_batch_blocks):
        for qkv_block_idx in range(num_qkv_blocks):
            instructions.append(
                QKV_MatMulRopeAppend(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    qkv_block_idx=qkv_block_idx,
                )
            )

    return instructions


def schedule_attention_decode(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(0, globs.batch_size):
        for kv_head_idx in range(globs.num_kv_heads):
            instructions.append(
                AttentionDecode(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    kv_head_idx=kv_head_idx,
                ),
            )
    return instructions


def schedule_o_proj_residual(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(globs.num_batch_blocks()):
        for o_block_idx in range(globs.num_output_blocks()):
            instructions.append(
                O_ProjResidual(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    output_block_idx=o_block_idx,
                )
            )

    return instructions


def schedule_pre_mlp_norm(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(0, globs.batch_size):
        instructions.append(
            PreMLP_Norm(
                layer_idx=layer_idx,
                batch_start_idx=bidx,
            )
        )

    return instructions


def schedule_gate_silu(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(globs.num_batch_blocks()):
        for block_idx in range(globs.num_intermediate_blocks()):
            instructions.append(
                GateSilu(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    output_block_idx=block_idx,
                )
            )

    return instructions


def schedule_up_matmul(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(globs.num_batch_blocks()):
        for block_idx in range(globs.num_intermediate_blocks()):
            instructions.append(
                UpMatMul(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    output_block_idx=block_idx,
                )
            )

    return instructions


def schedule_down_proj_residual(
    globs: Globals,
    layer_idx: int,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(globs.num_batch_blocks()):
        for down_block_idx in range(globs.num_output_blocks()):
            instructions.append(
                DownProjResidual(
                    layer_idx=layer_idx,
                    batch_start_idx=bidx,
                    output_block_idx=down_block_idx,
                )
            )
    return instructions


def schedule_lm_head_norm(
    globs: Globals,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(0, globs.batch_size):
        instructions.append(
            PreLMHeadRMS(
                layer_idx=0,
                batch_start_idx=bidx,
            )
        )

    return instructions


def schedule_lm_head(
    globs: Globals,
) -> tuple[list[Instruction], bool]:
    instructions: list[Instruction] = []

    for bidx in range(globs.num_batch_blocks()):
        for logit_block_idx in range(globs.num_vocab_blocks()):
            instructions.append(
                LM_Head(
                    batch_start_idx=bidx,
                    output_block_idx=logit_block_idx,
                )
            )

    return instructions


def make_dag(
    globs: Globals, stop_after_op: str | None = None, layer_limit: int | None = None
):
    nodes: list[DAG_Node] = []

    if layer_limit is not None:
        nlayers = layer_limit
    else:
        nlayers = globs.num_hidden_layers

    last_outputs = []
    for layer_idx in range(nlayers):
        new_nodes, new_outputs = make_dag_layer(
            globs=globs,
            layer_idx=layer_idx,
            prev_layer_outputs=last_outputs,
            stop_after_op=stop_after_op,
        )
        nodes.extend(new_nodes)
        last_outputs = new_outputs

    if nlayers == globs.num_hidden_layers:
        lm_head_norm_instructions = schedule_lm_head_norm(globs)
        lm_head_norm_nodes: list[DAG_Node] = []
        for ins in lm_head_norm_instructions:
            lm_head_norm_nodes.append(DAG_Node(ins, last_outputs))

        nodes.extend(lm_head_norm_nodes)
        last_outputs = lm_head_norm_nodes

        if stop_after_op != "lm_head_norm":
            lm_head_instructions = schedule_lm_head(globs)
            lm_head_nodes: list[DAG_Node] = []

            # TODO: these dependencies are broken bc
            # the lm head norm and matmul is split into two instructions
            for ins in lm_head_instructions:
                lm_head_nodes.append(DAG_Node(ins, last_outputs))

            nodes.extend(lm_head_nodes)
            last_outputs = lm_head_nodes

    end_node = DAG_Node(NoOp(), last_outputs)

    return nodes, end_node


def make_dag_layer(
    globs: Globals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    stop_after_op: str | None = None,
):
    new_nodes: list[DAG_Node] = []

    # INSTRUCTION 1: PreAttnNorm
    ins = schedule_pre_attn_norm(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "attn_norm":
        return new_nodes, []

    # INSTRUCTION 2: QKV_MatMulRopeAppend
    ins = schedule_qkv_matmul_rope_append(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "qkv":
        return new_nodes, []

    # INSTRUCTION 3: AttentionDecode
    # Need to figure out how to give in kv_indices
    ins = schedule_attention_decode(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "attn":
        return new_nodes, []

    # INSTRUCTION 4: O_ProjResidual
    ins = schedule_o_proj_residual(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "oproj":
        return new_nodes, []

    ###########################################
    # MLP
    ###########################################

    # INSTRUCTION 5: PreMLPNorm
    ins = schedule_pre_mlp_norm(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "mlp_norm":
        return new_nodes, []

    # INSTRUCTION 6: GateSilu
    ins = schedule_gate_silu(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "gate":
        return new_nodes, []

    # INSTRUCTION 7: UpMatMul
    ins = schedule_up_matmul(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    if stop_after_op == "up":
        return new_nodes, []

    # INSTRUCTION 8: DownProjResidual
    ins = schedule_down_proj_residual(globs, layer_idx)
    new_nodes.extend([DAG_Node(i, prev_layer_outputs) for i in ins])

    assert stop_after_op is None

    return new_nodes, []


class ThroughputScheduleBuilder(ScheduleBuilder):
    @classmethod
    def make_globals(cls, model):
        return make_globals(model)

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None
    ):
        return make_dag(globs, stop_after_op, layer_limit)
