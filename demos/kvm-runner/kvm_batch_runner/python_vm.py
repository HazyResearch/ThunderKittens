import sys
import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from kvm_batch_runner.instructions import (

    PreLayerNorm,
    QKV_MatMulRopeAppend,
    AttentionDecode,
    O_ProjResidual,
    
    PreMLPLayerNorm,
    GateSilu,
    UpMatMul,
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
from typing import Optional, Dict

include_barriers = False

def _store_debug_tensor(debug_outputs: Optional[Dict[str, Tensor]],
                        key: str,
                        tensor: Tensor,
                        clone_detach_cpu: bool = False):
    if debug_outputs is not None:
        if clone_detach_cpu:
            debug_outputs[key] = tensor.clone().detach().cpu()
        else:
            debug_outputs[key] = tensor.clone()

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matmul(
    matA: Tensor,
    matB: Tensor,
):
    out = einsum(matA, matB, "a i, b i -> a b" )
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
):    
    matmul_out = matmul(matA, matB)
    
    residual += matmul_out

def pre_attn_layer_norm(
    globals: Globals,
    instruction: PreLayerNorm,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx

    # First barrier check
    if include_barriers:
        if layer_idx > 0:
            op_barriers = globals.barriers[layer_idx - 1, batch_start_idx - 1, instruction.prev_opcode() - 1]
            assert op_barriers[0] == 64 # 8192 (hidden dim) / 128 (block size); from down_proj tiles

    pre_attn_ln = rms_norm(
        inp=globals.hidden_states[batch_start_idx],
        weight=globals.attn_ln_weight[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_rope_intermediates[batch_start_idx] = pre_attn_ln

    # barrier update
    if include_barriers:
        barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        barriers[0] += 1
    
    if batch_start_idx == 127:
        _store_debug_tensor(debug_outputs, f"L{layer_idx}_pre_attn_ln", globals.rms_rope_intermediates)
    
    # print(f"Done with pre layer norm for layer {layer_idx} and batch {batch_start_idx}")


def qkv_matmul_rope_append(
    globals: Globals,
    instruction: QKV_MatMulRopeAppend,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    qkv_block_idx = instruction.qkv_block_idx
    batch_end_idx = batch_start_idx + globals.batch_block_size
    batch_size = batch_end_idx - batch_start_idx

    # Barrier check
    if include_barriers:
        op_barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.prev_opcode() - 1]

    matmul_output = einsum(
        globals.qkv_proj[layer_idx], 
        globals.rms_rope_intermediates, "o i, b i -> b o"
    )

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    q = matmul_output[batch_start_idx:batch_end_idx, :k_start]
    k = matmul_output[batch_start_idx:batch_end_idx:, k_start:v_start]
    v = matmul_output[batch_start_idx:batch_end_idx:, v_start:]

    # print("Batch start idx: ", batch_start_idx)
    # print("Batch end idx: ", batch_end_idx)
    q_arr = rearrange(q, "b (h d) -> b h d", h=globals.num_attention_heads)
    k_arr = rearrange(k, "b (h d) -> b h d", h=globals.num_kv_heads)
    v_arr = rearrange(v, "b (h d) -> b h d", h=globals.num_kv_heads)

    pos_id = globals.pos_id
    q_with_rope, k_with_rope = apply_rotary_pos_emb_interleaved(
        q=q_arr,
        k=k_arr,
        cos=globals.rope_cos[pos_id],
        sin=globals.rope_sin[pos_id],
        unsqueeze_dim=0,
    )

    # Start and end here are global_head_idx * 128
    start, end = get_start_end(globals.qkv_block_size, qkv_block_idx)
    if start < k_start:
        globals.post_ln_rope_q[batch_start_idx:batch_end_idx, start:end] = q_with_rope.reshape(batch_size, -1)[:, start:end]
        if batch_start_idx == 0:
            debug_q_rope = rearrange(globals.post_ln_rope_q, "b (h d) -> b h d", h=globals.num_attention_heads)
            _store_debug_tensor(debug_outputs, f"L{layer_idx}_Q_rope", debug_q_rope)
    else:
       # Each batch needs to go to its own page
       for i in range(batch_size):
            block_idx = pos_id // 128
            slot_idx = pos_id % 128
            pid = globals.page_table[batch_start_idx + i][block_idx]

            if pid < 0:
                # Need to allocate a new page
                pid = globals.next_free_page.item()
                globals.page_table[batch_start_idx + i][block_idx] = pid
                globals.next_free_page[0] += 1
            
            # Determine which head to put the key/value in
            if start < v_start: # Key
                k_head_idx = (start - k_start) // globals.head_dim
                globals.k_cache[layer_idx, pid, k_head_idx, slot_idx, :] = k_with_rope[i, k_head_idx, :]
                _store_debug_tensor(debug_outputs, f"L{layer_idx}_B{i}_K_rope", globals.k_cache[layer_idx, pid, :, slot_idx, :])
            else: # Value
                v_head_idx = (start - v_start) // globals.head_dim
                globals.v_cache[layer_idx, pid, v_head_idx, slot_idx, :] = v_arr[i, v_head_idx, :]
                _store_debug_tensor(debug_outputs, f"L{layer_idx}_B{i}_V_arr", globals.v_cache[layer_idx, pid, :, slot_idx, :])

    # Barrier update
    if include_barriers:
        barriers = globals.barriers[instruction.layer_idx, instruction.batch_start_idx, instruction.opcode() - 1]
        barriers[qkv_block_idx // 8] += 1


def attention_decode(
    globals: Globals,
    instruction: AttentionDecode,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx       = instruction.layer_idx
    batch_idx       = instruction.batch_start_idx
    kv_head_idx     = instruction.kv_head_idx

    gqa_ratio   = globals.num_attention_heads // globals.num_kv_heads
    kv_head_dim = globals.hidden_size // globals.num_kv_heads

    if include_barriers:
        opb = globals.barriers[layer_idx, batch_idx, instruction.prev_opcode() - 1]
        for i in range(gqa_ratio):
            assert opb[kv_head_idx * gqa_ratio + i] == 8
        assert opb[globals.num_attention_heads + kv_head_idx] == 8
        assert opb[globals.num_attention_heads + globals.num_kv_heads + kv_head_idx] == 8

    # slice out Q for this head: shape [gqa_ratio, head_dim]
    q_flat      = globals.post_ln_rope_q[batch_idx]
    head_start  = kv_head_idx * gqa_ratio * globals.head_dim
    head_end    = head_start + gqa_ratio * globals.head_dim
    q           = q_flat[head_start:head_end].view(gqa_ratio, globals.head_dim).float()

    # how many pages × block_size we need
    num_kv_pages = math.ceil((globals.pos_id + 1) / globals.qkv_block_size)

    # initialize streaming accumulators (per‐head)
    device        = q.device
    running_max   = torch.full((gqa_ratio,), -float("inf"), device=device)
    running_denom = torch.zeros((gqa_ratio,),       device=device)
    running_out   = torch.zeros((gqa_ratio, globals.head_dim),
                                device=device)

    # loop over KV pages
    for page_idx in range(num_kv_pages):
        page = globals.page_table[batch_idx][page_idx]
        # k,v shape: [seq_block, head_dim]
        k = globals.k_cache[layer_idx, page, kv_head_idx, :, :].float()
        v = globals.v_cache[layer_idx, page, kv_head_idx, :, :].float()


        # 1) compute scaled scores for this block: [gqa_ratio, seq_block]
        qk         = torch.einsum("h d, s d -> h s", q, k)
        scaled_qk  = qk * globals.attn_scale

        # 2) streaming‐softmax update
        #   block_max: max per head over this page
        block_max = scaled_qk.amax(dim=1)                     # [gqa_ratio]
        new_max   = torch.maximum(running_max, block_max)     # [gqa_ratio]

        exp_old   = torch.exp(running_max - new_max)          # [gqa_ratio]
        exp_block = torch.exp(scaled_qk - new_max.unsqueeze(1))  # [gqa_ratio, seq_block]

        # numerator contribution: exp_block @ V  -> [gqa_ratio, head_dim]
        num_block   = torch.matmul(exp_block, v)
        running_out = running_out * exp_old.unsqueeze(1) + num_block

        # denominator contribution
        denom_block   = exp_block.sum(dim=1)                  # [gqa_ratio]
        running_denom = running_denom * exp_old + denom_block

        # commit new max
        running_max = new_max

    # 3) finalize: divide numerator by denominator
    out   = running_out / running_denom.unsqueeze(1)         # [gqa_ratio, head_dim]
    out_flat = out.view(-1)                                  # [gqa_ratio*head_dim]

    # store back
    out_start = kv_head_idx * kv_head_dim
    out_end   = out_start + kv_head_dim
    globals.attn_out[batch_idx, out_start:out_end] = out_flat

    debug_attn_out = rearrange(globals.attn_out[batch_idx], "(h d) -> h d", h=globals.num_attention_heads)
    _store_debug_tensor(debug_outputs, f"L{layer_idx}_B{batch_idx}_attn_out", debug_attn_out)

    if include_barriers:
        next_opb = globals.barriers[layer_idx, batch_idx, instruction.opcode() - 1]
        next_opb[0] += globals.attn_reduction_size

def o_proj_residual(
    globals: Globals,
    instruction: O_ProjResidual,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx       = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx   = batch_start_idx + globals.batch_block_size

    block_num  = instruction.output_block_idx
    block_size = globals.o_proj_block_size
    start      = block_num * block_size
    end        = start + block_size

    if include_barriers:
        op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == globals.num_attention_heads

    matmul_with_residual(
        matA     = globals.attn_out[batch_start_idx:batch_end_idx],      # [batch, hidden_size]
        matB     = globals.o_proj[layer_idx, start:end],                 # [block_size, hidden_size]
        residual = globals.hidden_states[batch_start_idx:batch_end_idx, start:end],  # [batch, block_size]
    )

    _store_debug_tensor(
        debug_outputs,
        f"L{layer_idx}_o_proj_residual",
        globals.hidden_states
    )

    if include_barriers:
        next_op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1

def pre_mlp_layer_norm(
    globals: Globals,
    instruction: PreMLPLayerNorm,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx

    if include_barriers:
        op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 128

    post_ln = rms_norm(
        inp=globals.hidden_states[batch_start_idx],
        weight=globals.mlp_ln_weight[layer_idx],
        eps=globals.rms_norm_eps,
    )

    globals.rms_gate_intermediates[batch_start_idx] = post_ln

    if batch_start_idx == 127:
        _store_debug_tensor(
            debug_outputs, 
            f"L{layer_idx}_pre_mlp_layer_norm",
            globals.rms_gate_intermediates
        )

    # barrier update
    if include_barriers:
        barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        barriers[0] += 1


def gate_silu(
    globals: Globals,
    instruction: GateSilu,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx       = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx   = batch_start_idx + globals.batch_block_size

    block_num  = instruction.output_block_idx
    block_size = globals.matmul_silu_block_size
    start      = block_num * block_size
    end        = start + block_size

    if include_barriers:
        op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == globals.hidden_size  # e.g. 8192

    # project into gate & apply SiLU
    matmul_out = matmul( # [B, intermediate]
        matA = globals.rms_gate_intermediates[batch_start_idx:batch_end_idx],  # [B, hidden]
        matB = globals.gate_proj[layer_idx, start:end],                        # [intermediate, hidden]
    )
    post_silu = F.silu(matmul_out)

    globals.silu_out[batch_start_idx:batch_end_idx, start:end] = post_silu

    _store_debug_tensor(
        debug_outputs,
        f"L{layer_idx}_gate_silu",
        globals.silu_out,
    )

    if include_barriers:
        next_op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


def up_matmul(
    globals: Globals,
    instruction: UpMatMul,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx       = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx   = batch_start_idx + globals.batch_block_size

    block_num  = instruction.output_block_idx
    block_size = globals.matmul_gate_block_size
    start      = block_num * block_size
    end        = start + block_size

    if include_barriers:
        op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == block_size

    # matmul up_proj  and * the SiLU gate
    matmul_out = matmul( # [B, intermediate]
        matA = globals.hidden_states[batch_start_idx:batch_end_idx],  # [B, hidden]
        matB = globals.up_proj[layer_idx, start:end],                # [intermediate, hidden]
    )
    gated = matmul_out * globals.silu_out[batch_start_idx:batch_end_idx, start:end]

    globals.silu_out[batch_start_idx:batch_end_idx, start:end] = gated

    _store_debug_tensor(
        debug_outputs,
        f"L{layer_idx}_up_matmul",
        globals.silu_out,
    )

    if include_barriers:
        next_op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1


def down_proj_residual(
    globals: Globals,
    instruction: DownProjResidual,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    layer_idx       = instruction.layer_idx
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx   = batch_start_idx + globals.batch_block_size

    block_num  = instruction.output_block_idx
    block_size = globals.down_proj_block_size  # 128
    start      = block_num * block_size
    end        = start + block_size

    if include_barriers:
        op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.prev_opcode() - 1]
        assert op_barriers[0] == globals.hidden_size  # e.g. 8192

    matmul_with_residual(
        matA     = globals.silu_out[batch_start_idx:batch_end_idx],      # [batch, intermediate_size]
        matB     = globals.down_proj[layer_idx, start:end],             # [block_size, intermediate_size]
        residual = globals.hidden_states[batch_start_idx:batch_end_idx, start:end],  # [batch, block_size]
    )

    _store_debug_tensor(
        debug_outputs,
        f"L{layer_idx}_down_proj_residual",
        globals.hidden_states
    )

    if include_barriers:
        next_op_barriers = globals.barriers[layer_idx, batch_start_idx, instruction.opcode() - 1]
        next_op_barriers[0] += 1

# TODO Split this out for the throughput setting into RMS and LM Head
def rms_lm_head(
    globals: Globals,
    instruction: RMS_LM_Head,
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    batch_start_idx = instruction.batch_start_idx
    batch_end_idx   = instruction.batch_end_idx

    block_num  = instruction.output_block_idx
    block_size = globals.lm_head_block_size  # e.g. 128
    start      = block_num * block_size
    end        = start + block_size

    if include_barriers:
        op_barriers = globals.barriers[
            globals.num_hidden_layers - 1,
            batch_start_idx - 1,
            instruction.prev_opcode() - 1
        ]
        assert op_barriers[0] == globals.hidden_size  

    post_ln = rms_norm(
        inp    = globals.hidden_states[batch_start_idx:batch_end_idx],
        weight = globals.lm_head_norm_weights,
        eps    = globals.rms_norm_eps,
    )

    matmul_output = matmul(
        post_ln,
        globals.lm_head_weights[start:end],  
    )

    globals.logits[batch_start_idx:batch_end_idx, start:end] = matmul_output

    _store_debug_tensor(
        debug_outputs,
        f"LMHead_block_{block_num}",
        matmul_output
    )

    if include_barriers:
        next_op_barriers = globals.barriers[
            globals.num_hidden_layers - 1,
            batch_start_idx - 1,
            instruction.opcode() - 1
        ]
        next_op_barriers[0] += 1

# def rms_lm_head(
#     globals: Globals,
#     instruction: RMS_LM_Head,
#     debug_outputs: Optional[Dict[str, Tensor]] = None,
# ):
#     batch_start_idx = instruction.batch_start_idx
#     batch_end_idx = instruction.batch_end_idx
    
#     # Barrier check
#     if include_barriers:
#         op_barriers = globals.barriers[globals.num_hidden_layers - 1, batch_start_idx - 1, instruction.prev_opcode() - 1]
#         assert op_barriers[0] == 512

#     post_ln = rms_norm(
#         inp=globals.hidden_states[batch_start_idx:batch_end_idx],
#         weight=globals.lm_head_norm_weights,
#         eps=globals.rms_norm_eps,
#     )

#     matmul_output = matmul(
#         post_ln,
#         globals.lm_head_weights,
#     )

#     globals.logits[batch_start_idx:batch_end_idx] = matmul_output


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

    PreMLPLayerNorm: pre_mlp_layer_norm,
    GateSilu: gate_silu,
    UpMatMul: up_matmul,
    DownProjResidual: down_proj_residual,
    
    RMS_LM_Head: rms_lm_head,
    PrintState: print_state,
}


def interpret_with_pyvm(
    globals: Globals,
    instructions: list[Instruction],
    debug_outputs: Optional[Dict[str, Tensor]] = None,
):
    for instruction in instructions:
        INSTRUCTION_TO_SOLVER[type(instruction)](globals, instruction, debug_outputs)


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

    def run(
        self,
        input_ids: Tensor,
        pos_id: int,
        debug_outputs: Optional[Dict[str, Tensor]] = None,
    ):
        batch_state = BatchState(
            input_ids=input_ids,
        )
        
        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.globals.hidden_states[:, ] = hiddens.squeeze(1)
        self.globals.barriers.zero_()
        self.globals.pos_id = pos_id

        interpret_with_pyvm(self.globals, self.instructions, debug_outputs)
        output_hiddens = self.globals.hidden_states
        post_embedding.hidden_states = output_hiddens
        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids




