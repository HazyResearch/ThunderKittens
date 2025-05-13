import math

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

USE_BARRIERS = True


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
    batch_start_idx = instruction.batch_start_idx

    # First barrier check
    if USE_BARRIERS:
        if layer_idx > 0:
            op_barriers = globals.barriers[
                layer_idx - 1, instruction.prev_opcode() - 1, batch_start_idx - 1
            ]
            assert (
                op_barriers[0] == 64
            )  # 8192 (hidden dim) / 128 (block size); from down_proj tiles

    pre_attn_ln = rms_norm(
        inp=globals.hidden_states[batch_start_idx],
        weight=globals.attn_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_rope_intermediates[batch_start_idx] = pre_attn_ln

    # barrier update
    if USE_BARRIERS:
        batch_block_idx = batch_start_idx // globals.matmul_batch_block_size
        globals.barriers[layer_idx, instruction.opcode() - 1, batch_block_idx] += 1


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

    # Barrier check
    if USE_BARRIERS:
        assert (
            globals.barriers[
                instruction.layer_idx,
                instruction.prev_opcode() - 1,
                instruction.batch_start_idx,
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

    num_generated_heads = assert_div(end - start, globals.head_dim)

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
    if USE_BARRIERS:
        start_bar = (
            instruction.batch_start_idx * globals.num_total_heads()
            + batch_start_row // globals.head_dim
        )
        end_bar = start_bar + num_generated_heads
        globals.barriers[
            instruction.layer_idx,
            instruction.opcode() - 1,
            start_bar:end_bar,
        ] += 1


def attention_decode(
    globals: Globals,
    instruction: AttentionDecode,
):
    layer_idx = instruction.layer_idx
    batch_idx = instruction.batch_start_idx
    kv_head_idx = instruction.kv_head_idx

    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads
    kv_head_dim = globals.hidden_size // globals.num_kv_heads

    if USE_BARRIERS:
        opb = globals.barriers[layer_idx, batch_idx, instruction.prev_opcode() - 1]
        for i in range(gqa_ratio):
            assert opb[kv_head_idx * gqa_ratio + i] == 8
        assert opb[globals.num_attention_heads + kv_head_idx] == 8
        assert (
            opb[globals.num_attention_heads + globals.num_kv_heads + kv_head_idx] == 8
        )

    # slice out Q for this head: shape [gqa_ratio, head_dim]
    q_flat = globals.post_ln_rope_q[batch_idx]
    head_start = kv_head_idx * gqa_ratio * globals.head_dim
    head_end = head_start + gqa_ratio * globals.head_dim
    q = q_flat[head_start:head_end].view(gqa_ratio, globals.head_dim).float()

    # how many pages × block_size we need
    num_kv_pages = math.ceil((globals.pos_id + 1) / globals.qkv_block_size)

    # initialize streaming accumulators (per‐head)
    device = q.device
    running_max = torch.full((gqa_ratio,), -float("inf"), device=device)
    running_denom = torch.zeros((gqa_ratio,), device=device)
    running_out = torch.zeros((gqa_ratio, globals.head_dim), device=device)

    # loop over KV pages
    for page_idx in range(num_kv_pages):
        page = globals.page_table[batch_idx][page_idx]
        # k,v shape: [seq_block, head_dim]
        k = globals.k_cache[layer_idx, page, kv_head_idx, :, :].float()
        v = globals.v_cache[layer_idx, page, kv_head_idx, :, :].float()

        # 1) compute scaled scores for this block: [gqa_ratio, seq_block]
        qk = torch.einsum("h d, s d -> h s", q, k)
        scaled_qk = qk * globals.attn_scale

        # 2) streaming‐softmax update
        #   block_max: max per head over this page
        block_max = scaled_qk.amax(dim=1)  # [gqa_ratio]
        new_max = torch.maximum(running_max, block_max)  # [gqa_ratio]

        exp_old = torch.exp(running_max - new_max)  # [gqa_ratio]
        exp_block = torch.exp(
            scaled_qk - new_max.unsqueeze(1)
        )  # [gqa_ratio, seq_block]

        # numerator contribution: exp_block @ V  -> [gqa_ratio, head_dim]
        num_block = torch.matmul(exp_block, v)
        running_out = running_out * exp_old.unsqueeze(1) + num_block

        # denominator contribution
        denom_block = exp_block.sum(dim=1)  # [gqa_ratio]
        running_denom = running_denom * exp_old + denom_block

        # commit new max
        running_max = new_max

    # 3) finalize: divide numerator by denominator
    out = running_out / running_denom.unsqueeze(1)  # [gqa_ratio, head_dim]
    out_flat = out.view(-1)  # [gqa_ratio*head_dim]

    # store back
    out_start = kv_head_idx * kv_head_dim
    out_end = out_start + kv_head_dim
    globals.attn_out[batch_idx, out_start:out_end] = out_flat

    if USE_BARRIERS:
        next_opb = globals.barriers[layer_idx, batch_idx, instruction.opcode() - 1]
        next_opb[0] += globals.attn_reduction_size


def o_proj_residual(
    globals: Globals,
    instruction: O_ProjResidual,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = batch_start_idx + globals.matmul_block_size

    block_num = instruction.output_block_idx
    block_size = globals.o_proj_block_size
    start = block_num * block_size
    end = start + block_size

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == globals.num_attention_heads

    matmul_with_residual(
        matA=globals.attn_out[batch_start_idx:batch_end_idx],  # [batch, hidden_size]
        matB=globals.o_proj_weights[layer_idx, start:end],  # [block_size, hidden_size]
        residual=globals.hidden_states[
            batch_start_idx:batch_end_idx, start:end
        ],  # [batch, block_size]
    )

    if USE_BARRIERS:
        next_op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1


def pre_mlp_layer_norm(
    globals: Globals,
    instruction: PreMLP_Norm,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == 128

    post_ln = rms_norm(
        inp=globals.hidden_states[batch_start_idx],
        weight=globals.mlp_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_gate_intermediates[batch_start_idx] = post_ln

    # barrier update
    if USE_BARRIERS:
        barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.opcode() - 1
        ]
        barriers[0] += 1


def gate_silu(
    globals: Globals,
    instruction: GateSilu,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = batch_start_idx + globals.matmul_block_size

    block_num = instruction.output_block_idx
    block_size = globals.matmul_silu_block_size
    start = block_num * block_size
    end = start + block_size

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == globals.hidden_size  # e.g. 8192

    # project into gate & apply SiLU
    matmul_out = matmul(  # [B, intermediate]
        matA=globals.rms_gate_intermediates[
            batch_start_idx:batch_end_idx
        ],  # [B, hidden]
        matB=globals.gate_proj_weights[layer_idx, start:end],  # [intermediate, hidden]
    )
    post_silu = F.silu(matmul_out)

    globals.silu_out[batch_start_idx:batch_end_idx, start:end] = post_silu

    if USE_BARRIERS:
        next_op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1


def up_matmul(
    globals: Globals,
    instruction: UpMatMul,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = batch_start_idx + globals.matmul_block_size

    block_num = instruction.output_block_idx
    block_size = globals.matmul_gate_block_size
    start = block_num * block_size
    end = start + block_size

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == block_size

    # matmul up_proj  and * the SiLU gate
    matmul_out = matmul(  # [B, intermediate]
        matA=globals.rms_gate_intermediates[
            batch_start_idx:batch_end_idx
        ],  # [B, hidden]
        matB=globals.up_proj_weights[layer_idx, start:end],  # [intermediate, hidden]
    )
    gated = matmul_out * globals.silu_out[batch_start_idx:batch_end_idx, start:end]

    globals.silu_out[batch_start_idx:batch_end_idx, start:end] = gated

    if USE_BARRIERS:
        next_op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1


def down_proj_residual(
    globals: Globals,
    instruction: DownProjResidual,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = batch_start_idx + globals.matmul_block_size

    block_num = instruction.output_block_idx
    block_size = globals.down_proj_block_size  # 128
    start = block_num * block_size
    end = start + block_size

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == globals.hidden_size  # e.g. 8192

    matmul_with_residual(
        matA=globals.silu_out[
            batch_start_idx:batch_end_idx
        ],  # [batch, intermediate_size]
        matB=globals.down_proj_weights[
            layer_idx, start:end
        ],  # [block_size, intermediate_size]
        residual=globals.hidden_states[
            batch_start_idx:batch_end_idx, start:end
        ],  # [batch, block_size]
    )

    if USE_BARRIERS:
        next_op_barriers = globals.barriers[
            layer_idx, batch_start_idx, instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1


def pre_lm_head_rms(
    globals: Globals,
    instruction: PreLMHeadRMS,
):
    batch_start_idx = instruction.batch_start_idx

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            79, batch_start_idx, instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == 128

    post_ln = rms_norm(
        inp=globals.hidden_states[batch_start_idx],
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    globals.rms_lm_head_intermediates[batch_start_idx] = post_ln

    if USE_BARRIERS:
        barriers = globals.barriers[79, batch_start_idx, instruction.opcode() - 1]
        barriers[0] += 1


# TODO Split this out for the throughput setting into RMS and LM Head
def lm_head(
    globals: Globals,
    instruction: LM_Head,
):
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = batch_start_idx + globals.matmul_block_size

    block_num = instruction.output_block_idx
    block_size = globals.lm_head_block_size  # e.g. 128
    start = block_num * block_size
    end = start + block_size

    if USE_BARRIERS:
        op_barriers = globals.barriers[
            globals.num_hidden_layers - 1,
            batch_start_idx - 1,
            instruction.prev_opcode() - 1,
        ]
        assert op_barriers[0] == globals.hidden_size

    matmul_output = matmul(
        globals.rms_lm_head_intermediates,
        globals.lm_head_weights[start:end],
    )

    globals.logits[batch_start_idx:batch_end_idx, start:end] = matmul_output

    if USE_BARRIERS:
        next_op_barriers = globals.barriers[
            globals.num_hidden_layers - 1, batch_start_idx - 1, instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1


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
