import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from kvm_unity.llama import (
    apply_rotary_pos_emb,
)
from kvm_unity.python_vm import get_start_end, rms_norm
from kvm_unity.throughput.instructions import (
    AttentionDecode,
    DownProjResidual,
    GateSilu,
    Globals,
    LM_Head,
    O_ProjResidual,
    PreAttnLayerNorm,
    PreLMHeadRMS,
    PreMLP_Norm,
    QKV_MatMulRopeAppend,
    UpMatMul,
)
from kvm_unity.utils import assert_div
from torch import Tensor


def matmul(
    matA: Tensor,
    matB: Tensor,
):
    out = einsum(matA, matB, "a i, b i -> a b")
    return out


def matmul_with_residual(
    matA: Tensor,
    matB: Tensor,
    residual: Tensor,
):
    matmul_out = matmul(matA, matB)

    residual += matmul_out


def pre_attn_layer_norm(
    globals: Globals,
    instruction: PreAttnLayerNorm,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    batch_block_idx = batch_idx // globals.matmul_batch_block_size

    if layer_idx > 0:
        assert (
            globals.barriers[
                layer_idx - 1, instruction.prev_opcode() - 1, batch_block_idx, 0
            ]
            == globals.num_output_blocks()
        )

    pre_attn_ln = rms_norm(
        inp=globals.hidden_states[batch_idx],
        weight=globals.attn_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_rope_intermediates[batch_idx] = pre_attn_ln

    # barrier update
    batch_block_idx = batch_idx // globals.matmul_batch_block_size
    globals.barriers[layer_idx, instruction.opcode() - 1, batch_block_idx, 0] += 1


def qkv_matmul_rope_append(
    globals: Globals,
    instruction: QKV_MatMulRopeAppend,
):
    layer_idx = instruction.layer_idx
    batch_start_row = instruction.batch_start_idx * globals.matmul_batch_block_size
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_block_idx = instruction.qkv_block_idx
    output_start_col = output_block_idx * globals.matmul_output_block_size
    output_end_col = output_start_col + globals.matmul_output_block_size

    pos_id = globals.pos_id

    assert (
        globals.barriers[
            instruction.layer_idx,
            instruction.prev_opcode() - 1,
            instruction.batch_start_idx,
            0,
        ]
        == globals.matmul_batch_block_size
    )

    matmul_output = einsum(
        globals.qkv_proj_weights[layer_idx, output_start_col:output_end_col],
        globals.rms_rope_intermediates[batch_start_row:batch_end_row],
        "o i, b i -> b o",
    )

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    start, end = get_start_end(globals.matmul_output_block_size, output_block_idx)

    if start < k_start:
        mode = "q"
    elif start < v_start:
        mode = "k"
    else:
        mode = "v"

    output = matmul_output

    if mode in "qk":
        arr = rearrange(output, "... (h d) -> ... h d", d=globals.head_dim)

        # not interleaved for big-batch version
        with_rope, _ = apply_rotary_pos_emb(
            q=arr,
            k=arr,
            cos=globals.rope_cos[pos_id],
            sin=globals.rope_sin[pos_id],
            unsqueeze_dim=-2,
        )

        output = rearrange(with_rope, "... h d -> ... (h d)")

    num_generated_heads = assert_div(globals.matmul_output_block_size, globals.head_dim)

    if mode == "q":
        assert end <= k_start
        globals.post_ln_rope_q[batch_start_row:batch_end_row, start:end] = output

    else:
        arr = rearrange(output, "... (h d) -> ... h d", d=globals.head_dim)

        # Determine which head to put the key/value in
        if mode == "k":  # Key
            k_head_idx = assert_div(start - k_start, globals.head_dim)
            globals.k_cache[
                layer_idx,
                batch_start_row:batch_end_row,
                pos_id,
                k_head_idx : k_head_idx + num_generated_heads,
            ] = arr
        else:  # Value
            v_head_idx = (start - v_start) // globals.head_dim
            globals.v_cache[
                layer_idx,
                batch_start_row:batch_end_row,
                pos_id,
                v_head_idx : v_head_idx + num_generated_heads,
            ] = arr

    # Barrier update
    start_bar = output_start_col // globals.head_dim
    end_bar = start_bar + num_generated_heads

    # the asserts are checking 1, so we can assign and not increment
    globals.barriers[
        instruction.layer_idx,
        instruction.opcode() - 1,
        instruction.batch_start_idx,
        start_bar:end_bar,
    ] = 1


def attention_decode(
    globals: Globals,
    instruction: AttentionDecode,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    kv_head_idx = instruction.kv_head_idx

    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads
    seq_len = globals.pos_id + 1

    q_head_start, q_head_end = get_start_end(gqa_ratio, kv_head_idx)

    # barrier check
    bar_idx = batch_idx // globals.matmul_batch_block_size
    bars = globals.barriers[layer_idx, instruction.prev_opcode() - 1, bar_idx]
    for i in range(gqa_ratio):
        assert bars[kv_head_idx * gqa_ratio + i] == 1
    assert bars[globals.num_attention_heads + kv_head_idx] == 1
    assert bars[globals.num_attention_heads + globals.num_kv_heads + kv_head_idx] == 1

    all_q = rearrange(
        globals.post_ln_rope_q[batch_idx],
        "(h d) -> h d",
        d=globals.head_dim,
    )

    q = all_q[q_head_start:q_head_end]

    k = globals.k_cache[layer_idx, batch_idx, :seq_len, kv_head_idx]
    v = globals.v_cache[layer_idx, batch_idx, :seq_len, kv_head_idx]

    qk = einsum(q.float(), k.float(), "qheads i, seqlen i -> qheads seqlen")
    scaled_qk = qk * globals.attn_scale

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    # lse = torch.logsumexp(scaled_qk, dim=-1)
    # lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "qheads seqlen, seqlen o -> qheads o")

    all_out = rearrange(globals.attn_out[batch_idx], "(h d) -> h d", d=globals.head_dim)
    all_out[q_head_start:q_head_end] = out

    globals.barriers[layer_idx, instruction.opcode() - 1, bar_idx, 0] += 1


def o_proj_residual(
    globals: Globals,
    instruction: O_ProjResidual,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    batch_start_row, batch_end_row = get_start_end(
        globals.matmul_batch_block_size, batch_idx
    )
    output_start_col, output_end_col = get_start_end(
        globals.matmul_output_block_size, instruction.output_block_idx
    )

    assert (
        globals.barriers[layer_idx, instruction.prev_opcode() - 1, batch_idx, 0]
        == globals.matmul_batch_block_size * globals.num_kv_heads
    )

    matmul_with_residual(
        matA=globals.attn_out[batch_start_row:batch_end_row],  # [batch, hidden_size]
        matB=globals.o_proj_weights[
            layer_idx, output_start_col:output_end_col
        ],  # [block_size, hidden_size]
        residual=globals.hidden_states[
            batch_start_row:batch_end_row, output_start_col:output_end_col
        ],  # [batch, block_size]
    )

    globals.barriers[layer_idx, instruction.opcode() - 1, batch_idx, 0] += 1


def pre_mlp_layer_norm(
    globals: Globals,
    instruction: PreMLP_Norm,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    batch_block_idx = batch_idx // globals.matmul_batch_block_size

    assert (
        globals.barriers[layer_idx, instruction.prev_opcode() - 1, batch_block_idx, 0]
        == globals.num_output_blocks()
    )

    post_norm = rms_norm(
        inp=globals.hidden_states[batch_idx],
        weight=globals.mlp_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_gate_intermediates[batch_idx] = post_norm

    # barrier update
    batch_block_idx = batch_idx // globals.matmul_batch_block_size
    globals.barriers[layer_idx, instruction.opcode() - 1, batch_block_idx, 0] += 1


def gate_silu(
    globals: Globals,
    instruction: GateSilu,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    out_idx = instruction.output_block_idx
    batch_start_row, batch_end_row = get_start_end(
        globals.matmul_batch_block_size, batch_idx
    )
    output_start_col, output_end_col = get_start_end(
        globals.matmul_output_block_size, out_idx
    )

    assert (
        globals.barriers[layer_idx, instruction.prev_opcode() - 1, batch_idx, 0]
        == globals.matmul_batch_block_size
    )

    # project into gate & apply SiLU
    matmul_out = matmul(  # [B, intermediate]
        matA=globals.rms_gate_intermediates[
            batch_start_row:batch_end_row
        ],  # [B, hidden]
        matB=globals.gate_proj_weights[
            layer_idx, output_start_col:output_end_col
        ],  # [intermediate, hidden]
    )
    post_silu = F.silu(matmul_out)

    globals.silu_out[batch_start_row:batch_end_row, output_start_col:output_end_col] = (
        post_silu
    )

    globals.barriers[layer_idx, instruction.opcode() - 1, batch_idx, out_idx] += 1


def up_matmul(
    globals: Globals,
    instruction: UpMatMul,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    out_idx = instruction.output_block_idx
    batch_start_row, batch_end_row = get_start_end(
        globals.matmul_batch_block_size, batch_idx
    )
    output_start_col, output_end_col = get_start_end(
        globals.matmul_output_block_size, out_idx
    )

    assert (
        globals.barriers[layer_idx, instruction.prev_opcode() - 1, batch_idx, out_idx]
        == 1
    )

    # matmul up_proj  and * the SiLU gate
    matmul_out = matmul(  # [B, intermediate]
        matA=globals.rms_gate_intermediates[
            batch_start_row:batch_end_row
        ],  # [B, hidden]
        matB=globals.up_proj_weights[
            layer_idx, output_start_col:output_end_col
        ],  # [intermediate, hidden]
    )
    gated = (
        matmul_out
        * globals.silu_out[
            batch_start_row:batch_end_row, output_start_col:output_end_col
        ]
    )

    globals.silu_out[batch_start_row:batch_end_row, output_start_col:output_end_col] = (
        gated
    )

    globals.barriers[layer_idx, instruction.opcode() - 1, batch_idx, 0] += 1


def down_proj_residual(
    globals: Globals,
    instruction: DownProjResidual,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    out_idx = instruction.output_block_idx
    batch_start_row, batch_end_row = get_start_end(
        globals.matmul_batch_block_size, batch_idx
    )
    output_start_col, output_end_col = get_start_end(
        globals.matmul_output_block_size, out_idx
    )

    assert globals.barriers[
        layer_idx, instruction.prev_opcode() - 1, batch_idx, 0
    ] == assert_div(globals.intermediate_size, globals.matmul_output_block_size)

    matmul_with_residual(
        matA=globals.silu_out[
            batch_start_row:batch_end_row
        ],  # [batch, intermediate_size]
        matB=globals.down_proj_weights[
            layer_idx, output_start_col:output_end_col
        ],  # [block_size, intermediate_size]
        residual=globals.hidden_states[
            batch_start_row:batch_end_row, output_start_col:output_end_col
        ],  # [batch, block_size]
    )

    globals.barriers[layer_idx, instruction.opcode() - 1, batch_idx, 0] += 1


def pre_lm_head_rms(
    globals: Globals,
    instruction: PreLMHeadRMS,
):
    batch_idx = instruction.batch_start_idx
    batch_block_idx = batch_idx // globals.matmul_batch_block_size

    assert (
        globals.barriers[
            globals.num_hidden_layers - 1,
            instruction.prev_opcode() - 1,
            batch_block_idx,
            0,
        ]
        == globals.num_output_blocks()
    )

    post_ln = rms_norm(
        inp=globals.hidden_states[batch_idx],
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    globals.rms_lm_head_intermediates[batch_idx] = post_ln

    globals.barriers[
        globals.num_hidden_layers - 1,
        instruction.opcode() - 1,
        batch_block_idx,
        0,
    ] += 1


# TODO Split this out for the throughput setting into RMS and LM Head
def lm_head(
    globals: Globals,
    instruction: LM_Head,
):
    batch_idx = instruction.batch_start_idx
    out_idx = instruction.output_block_idx
    batch_start_row, batch_end_row = get_start_end(
        globals.matmul_batch_block_size, batch_idx
    )
    output_start_col, output_end_col = get_start_end(
        globals.matmul_output_block_size, out_idx
    )

    assert (
        globals.barriers[
            globals.num_hidden_layers - 1, instruction.prev_opcode() - 1, batch_idx, 0
        ]
        == globals.matmul_batch_block_size
    )

    matmul_output = matmul(
        globals.rms_lm_head_intermediates[batch_start_row:batch_end_row],
        globals.lm_head_weights[output_start_col:output_end_col],
    )

    globals.logits[batch_start_row:batch_end_row, output_start_col:output_end_col] = (
        matmul_output
    )


INSTRUCTION_TO_SOLVER = {
    PreAttnLayerNorm: pre_attn_layer_norm,
    QKV_MatMulRopeAppend: qkv_matmul_rope_append,
    AttentionDecode: attention_decode,
    O_ProjResidual: o_proj_residual,
    PreMLP_Norm: pre_mlp_layer_norm,
    GateSilu: gate_silu,
    UpMatMul: up_matmul,
    DownProjResidual: down_proj_residual,
    PreLMHeadRMS: pre_lm_head_rms,
    LM_Head: lm_head,
}
