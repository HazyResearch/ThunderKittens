import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from kvm_batch_runner.instructions import (

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

    Globals,
    Instruction,
)
from kvm_batch_runner.llama import (
    BatchState,
    LlamaForCausalLM,
    apply_rotary_pos_emb_interleaved,
)
from kvm_batch_runner.scheduler import PrintInfo, schedule_model, tensorize_instructions
from kvm_batch_runner.utils import trepr
from torch import Tensor


include_barriers = False

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
    residual: Tensor,
    block_size: int,
    block_idx: int,
    # reduction_size: int,
    # reduction_block_idx: int,
):
    matmul_out, start, end = matmul(
        matA, vecB #, block_size, block_idx, True, reduction_size, reduction_block_idx
    )
    residual[start:end] += matmul_out


def pre_attn_layer_norm(
    globals: Globals, instruction: PreLayerNorm
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx

    # First barrier check
    if include_barriers:
        if layer_idx > 0:
            op_barriers = globals.barriers[layer_idx - 1, batch_start_idx - 1, instruction.prev_opcode() - 1]
            assert op_barriers[0] == 64 # 8192 (hidden dim) / 128 (block size); from down_proj tiles

    pre_attn_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weight[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.hidden_states[:] = pre_attn_ln

    # barrier update
    if include_barriers:
        barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        barriers[0] += 1

    print(f"Done with pre layer norm for layer {layer_idx} and batch {batch_start_idx}")


def qkv_matmul_rope_append(
    globals: Globals, instruction: QKV_MatMulRopeAppend
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx
    batch_size = batch_end_idx - batch_start_idx

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]

    matmul_output = einsum(
        globals.qkv_proj[layer_idx], 
        globals.hidden_states, "o i, b i -> b o"
    )

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    q = matmul_output[batch_start_idx:batch_end_idx, :k_start]
    k = matmul_output[batch_start_idx:batch_end_idx:, k_start:v_start]
    v = matmul_output[batch_start_idx:batch_end_idx:, v_start:]

    q_arr = rearrange(q, "b (h d) -> b h d", h=globals.num_attention_heads)
    k_arr = rearrange(k, "b (h d) -> b h d", h=globals.num_kv_heads)

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
        globals.post_ln_rope_q[batch_start_idx:batch_end_idx, start:end] = q_with_rope.view(batch_size, -1)[:, start:end]
    elif start < v_start:
        start_in_k = start - k_start
        end_in_k = end - k_start
        globals.k_cache[layer_idx, batch_start_idx:batch_end_idx, pos_id].view(batch_size, -1)[start_in_k:end_in_k] = ( k_with_rope.view(batch_size, -1)[start_in_k:end_in_k] )
    else:
        start_in_v = start - v_start
        end_in_v = end - v_start
        globals.v_cache[layer_idx, batch_start_idx:batch_end_idx, pos_id].view(batch_size, -1)[start_in_v:end_in_v] = v.view(batch_size, -1)[ start_in_v:end_in_v ]

    # Barrier update
    if include_barriers:
        barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        barriers[instruction.output_block_idx // 4] += 1


def attention_decode(globals: Globals, instruction: AttentionDecode):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx
    batch_size = batch_end_idx - batch_start_idx

    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads
    kv_head_dim = globals.hidden_size // globals.num_kv_heads

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
        for i in range(gqa_ratio):
            assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 4
        assert op_barriers[globals.num_attention_heads + instruction.kv_head_idx] == 4
        assert (op_barriers[globals.num_attention_heads + globals.num_kv_heads + instruction.kv_head_idx] == 4)

    kv_block_size = globals.attn_kv_block_size
    seq_len = globals.pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    k = globals.k_cache[layer_idx, batch_start_idx:batch_end_idx, :, kv_head_idx]
    v = globals.v_cache[layer_idx, batch_start_idx:batch_end_idx, :, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.post_ln_rope_q[batch_start_idx:batch_end_idx].view(batch_size, globals.num_attention_heads, -1)[:, head_start:head_end]

    try:
        qk = einsum(q.float(), k.float(), "b h i, b k i -> b h k")
    except:
        breakpoint()
    scaled_qk = qk * globals.attn_scale

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    out = einsum(softmax.float(), v.float(), "b h k, b k o -> b h o")
    
    globals.attn_out[batch_start_idx:batch_end_idx, kv_head_idx*kv_head_dim: (kv_head_idx+1)*kv_head_dim] = out.view(batch_size, -1)

    # Barrier update
    if include_barriers:
        next_op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += globals.attn_reduction_size  # the dumb way


def pre_mlp_layer_norm(
    globals: Globals, instruction: PostLayerNorm
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx

    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 128

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weight[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.hidden_states[:] = post_ln

    # barrier update
    if include_barriers:
        barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        barriers[0] += 1


def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == globals.num_attention_heads

    matmul_with_residual(
        matA=globals.o_proj[instruction.layer_idx],
        matB=globals.attn_out,
        residual=globals.hidden_states,
        # block_size=globals.o_proj_block_size,
        # block_idx=instruction.output_block_idx,
        # reduction_size=globals.matvec_reduction_size,
        # reduction_block_idx=instruction.reduction_block_idx,
    )

    # Barrier update
    if include_barriers:
        next_op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


def matmul_silu(globals: Globals, instruction: MatMulSiLU):
    
    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == globals.hidden_size

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
    if include_barriers:
        next_op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


def matmul_gate( globals: Globals, instruction: MatMulGate):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
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
    if include_barriers:
        next_op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


def down_proj_residual(globals: Globals, instruction: DownProjResidual):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx = instruction.batch_end_idx

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 512  # 8192 / 16

    matmul_with_residual(
        mat=globals.down_proj[instruction.layer_idx],
        vec=globals.silu_out,
        residual=globals.hidden_states,
        block_size=globals.down_proj_block_size,
        block_idx=instruction.output_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    if include_barriers:
        next_op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


# TODO Split this out for the throughput setting into RMS and LM Head
def rms_lm_head(globals: Globals, instruction: RMS_LM_Head):
    
    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[globals.num_hidden_layers - 1, instruction.batch_start_idx - 1, instruction.prev_opcode() - 1]
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
        "b o i, b i -> b o",
    )

    globals.logits[start:end] = matmul_output



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
    PreLayerNorm: pre_attn_layer_norm,

    QKV_MatMulRopeAppend: qkv_matmul_rope_append,
    AttentionDecode: attention_decode,
    O_ProjResidual: o_proj_residual,

    PostLayerNorm: pre_mlp_layer_norm,
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
        batch_size = self.model.extra_config.max_batch_size
        tensorize_instructions(self.globals, self.instructions, batch_size)

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )
        
        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.globals.hidden_states[:, ] = hiddens.squeeze(1)
        self.globals.barriers.zero_()
        self.globals.pos_id = pos_id

        interpret_with_pyvm(self.globals, self.instructions)
        output_hiddens = self.globals.hidden_states
        post_embedding.hidden_states = output_hiddens
        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids




