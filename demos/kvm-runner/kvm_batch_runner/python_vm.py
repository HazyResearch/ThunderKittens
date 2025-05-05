import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from kvm_runner.instructions import (
    PreLayerNorm,
    QKV_MatMulRopeAppend,
    AttentionDecode,
    O_ProjResidual,
    PostLayerNorm,
    MatMulSiLU,
    MatMulGate,
    DownProjResidual,
    PrintState,
    RMS_LM_Head,
)
from kvm_runner.llama import (
    BatchState,
    LlamaForCausalLM,
    apply_rotary_pos_emb_interleaved,
)
from kvm_runner.scheduler import PrintInfo, schedule_model, tensorize_instructions
from kvm_runner.utils import trepr
from torch import Tensor


def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matmul(
    matA: Tensor,
    matB: Tensor,
    # block_size: int,
    # block_idx: int,
    # reduce: bool = False,
    # reduction_size: int = 0,
    # reduction_idx: int = 0,
):
    out = einsum(matA, matB, "o i, i -> o")
    return out


def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def matmul_with_residual(
    matA: Tensor,
    matB: Tensor,
    # residual: Tensor,
    # block_size: int,
    # block_idx: int,
    # reduction_size: int,
    # reduction_block_idx: int,
):
    matmul_out, start, end = matmul(
        matA, vecB #, block_size, block_idx, True, reduction_size, reduction_block_idx
    )
    residual[start:end] += matmul_out


def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == globals.num_attention_heads

    matmul_with_residual(
        matA=globals.o_proj[instruction.layer_idx],
        matB=globals.attn_out,
        # residual=globals.hidden_states,
        # block_size=globals.o_proj_block_size,
        # block_idx=instruction.output_block_idx,
        # reduction_size=globals.matvec_reduction_size,
        # reduction_block_idx=instruction.reduction_block_idx,
    )

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += 1


def down_proj_residual(globals: Globals, instruction: DownProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 512  # 8192 / 16

    matvec_with_residual(
        mat=globals.down_proj[instruction.layer_idx],
        vec=globals.silu_out,
        residual=globals.hidden_states,
        block_size=globals.down_proj_block_size,
        block_idx=instruction.output_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += 1


def pre_layer_norm(
    globals: Globals, instruction: PreLayerNorm
):
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 128

    pre_attn_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weight[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.hidden_states[:] = pre_attn_ln


def post_layer_norm(
    globals: Globals, instruction: PostLayerNorm
):
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 128

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weight[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.hidden_states[:] = post_ln


def matmul_silu(
    globals: Globals, instruction: MatMulSiLU
):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 128

    block_size = globals.up_gate_proj_block_size

    gate_matmul, _, _ = matmul(
        matA=globals.gate_proj[instruction.layer_idx],
        matB=post_ln,
        # block_size=block_size,
        # block_idx=instruction.output_block_idx,
    )

    post_silu = F.silu(gate_matmul)

    globals.silu_out[start:end] = post_silu

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += 1


def matmul_gate(
    globals: Globals, instruction: MatMulGate
):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 128

    block_size = globals.up_gate_proj_block_size

    up_matmul, start, end = matmul(
        matA=globals.up_proj[instruction.layer_idx],
        matB=post_ln,
        # block_size=block_size,
        # block_idx=instruction.output_block_idx,
    ) 

    globals.silu_out = globals.silu_out * up_matmul

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += 1


def rms_lm_head(globals: Globals, instruction: RMS_LM_Head):
    op_barriers = globals.barriers[
        globals.num_hidden_layers - 1, instruction.prev_opcode() - 1
    ]
    assert op_barriers[0] == 512

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    start, end = get_start_end(globals.lm_head_block_size, instruction.output_block_idx)

    matmul_output = einsum(
        globals.lm_head_weights[start:end],
        post_ln,
        "o i, i -> o",
    )

    globals.logits[start:end] = matmul_output


def qkv_matmul_rope_append(
    globals: Globals, instruction: QKV_MatMulRopeAppend
):
    layer_idx = instruction.layer_idx

    # Barrier check
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 512

    matmul_output = einsum(
        globals.qkv_proj[layer_idx],
        globals.hidden_states,
        "o i, i -> o",
    )

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    q = matmul_output[:k_start]
    k = matmul_output[k_start:v_start]
    v = matmul_output[v_start:]

    q_arr = rearrange(q, "(h d) -> h d", h=globals.num_attention_heads)
    k_arr = rearrange(k, "(h d) -> h d", h=globals.num_kv_heads)

    pos_id = globals.pos_id
    q_with_rope, k_with_rope = apply_rotary_pos_emb_interleaved(
        q=q_arr,
        k=k_arr,
        cos=globals.rope_cos[pos_id],
        sin=globals.rope_sin[pos_id],
        unsqueeze_dim=0,
    )

    start, end = get_start_end(globals.qkv_block_size, instruction.output_block_idx)
    if start < k_start:
        globals.post_ln_rope_q[start:end] = q_with_rope.view(-1)[start:end]
    elif start < v_start:
        start_in_k = start - k_start
        end_in_k = end - k_start
        globals.k_cache[layer_idx, pos_id].view(-1)[start_in_k:end_in_k] = (
            k_with_rope.view(-1)[start_in_k:end_in_k]
        )
    else:
        start_in_v = start - v_start
        end_in_v = end - v_start
        globals.v_cache[layer_idx, pos_id].view(-1)[start_in_v:end_in_v] = v.view(-1)[
            start_in_v:end_in_v
        ]

    # Barrier update
    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    barriers[instruction.output_block_idx // 4] += 1


def attention_decode(globals: Globals, instruction: AttentionDecode):
    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 4
    assert op_barriers[globals.num_attention_heads + instruction.kv_head_idx] == 4
    assert (
        op_barriers[
            globals.num_attention_heads + globals.num_kv_heads + instruction.kv_head_idx
        ]
        == 4
    )

    kv_block_size = globals.attn_kv_block_size
    seq_len = globals.pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.post_ln_rope_q.view(globals.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * globals.attn_scale

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    # lse = torch.logsumexp(scaled_qk, dim=-1)
    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")

    globals.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = lse
    globals.attn_out_intermediates[head_start:head_end, instruction.partial_idx] = out
    
    ## OLD REDUCTION CODE ##

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    )

    lses = globals.attn_lse_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]
    outs = globals.attn_out_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values

    # adjusted_factors = (lses - max_lse).exp()
    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)

    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start_idx : head_start_idx + globals.attn_reduction_size
        ] = reduced
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = new_lse
        globals.attn_out_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = reduced

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += globals.attn_reduction_size  # the dumb way




def print_state(globals: Globals, instruction: PrintState):
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


INSTRUCTION_TO_SOLVER = {
    PreLayerNorm: layer_norm,

    QKV_MatMulRopeAppend: layer_norm_matvec_rope_append,
    AttentionDecode: attention,
    O_ProjResidual: o_proj_residual,

    PostLayerNorm: post_layer_norm,
    MatMulSiLU: matmul_silu,
    MatMulGate: matmul_gate,
    DownProjResidual: down_proj_residual,
    
    RMS_LM_Head: rms_lm_head,
    PrintState: print_state,
}


def interpret_with_pyvm(globals: Globals, instructions: list[Instruction]):
    for instruction in instructions:
        INSTRUCTION_TO_SOLVER[type(instruction)](globals, instruction)


class PyVM_Runner:
    def __init__(
        self,
        model: LlamaForCausalLM,
        prompt_len: int,
        ntok: int,
        print_info: PrintInfo | None = None,
    ):
        self.model = model

        self.globals, self.instructions = schedule_model(
            prompt_len=prompt_len,
            ntok=ntok,
            print_info=print_info,
            model=self.model,
        )

        tensorize_instructions(self.globals, self.instructions)

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )

        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.globals.hidden_states[:] = hiddens
        self.globals.barriers.zero_()
        self.globals.pos_id = pos_id

        interpret_with_pyvm(self.globals, self.instructions)

        output_hiddens = self.globals.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids




