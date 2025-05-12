from dataclasses import dataclass
from typing import Optional

from kvm_unity.instructions import BaseGlobals, Instruction
from torch import Tensor


@dataclass
class Globals(BaseGlobals):
    # activation buffers
    hidden_states: Tensor
    post_ln_rope_q: Tensor
    attn_out: Tensor
    attn_lse_intermediates: Tensor
    attn_out_intermediates: Tensor
    silu_out: Tensor
    logits: Tensor

    pos_id: int
    skip_attn_reduction: bool

    # block size constants
    up_gate_proj_block_size: int
    down_proj_block_size: int
    o_proj_block_size: int
    lm_head_block_size: int
    matvec_reduction_size: int
    qkv_block_size: int
    attn_kv_block_size: int
    attn_reduction_size: int


@dataclass
class LayerNorm_QKV_MatVecRopeAppend(Instruction):
    """
    attention: layernorm + qkv matvec + rope on q and k + append k and v to kv cache
    """

    layer_idx: int
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()

    def block_indices(self):
        return list(range(self.start_output_block_idx, self.end_output_block_idx))

    def cost(self, globs: Globals):
        return (
            (self.end_output_block_idx - self.start_output_block_idx)
            * globs.qkv_block_size
            * globs.hidden_size
        )


@dataclass
class PartialAttention(Instruction):
    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return LayerNorm_QKV_MatVecRopeAppend.opcode()

    def cost(self, globs: Globals):
        seq_len = globs.pos_id + 1
        loaded_seq_len = seq_len / self.num_partials

        # num loaded elements from kv cache
        return loaded_seq_len * globs.head_dim * 2


@dataclass
class AttentionReduction(Instruction):
    layer_idx: int
    head_start_idx: int
    # the original number of attention partitions
    num_partials: int
    is_terminal: bool
    # TODO: make sure reduction_list can't go beyond instruction
    reduction_list: list[int]
    # Not required for the last reduction
    output_partial_idx: Optional[int] = None

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return PartialAttention.opcode()


@dataclass
class MatVecAdd(Instruction):
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int


# denoting these with separate opcodes so that know what inputs to read from


@dataclass
class O_ProjResidual(MatVecAdd):
    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> int:
        return AttentionReduction.opcode()

    def cost(self, globs: Globals):
        return (
            (self.end_block_idx - self.start_block_idx)
            * globs.o_proj_block_size
            * globs.hidden_size
        )


@dataclass
class LayerNormDoubleMatVecSiLU(Instruction):
    """
    layernorm + double matvec + silu
    """

    layer_idx: int
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return O_ProjResidual.opcode()

    def cost(self, globs: Globals):
        return (
            (self.end_output_block_idx - self.start_output_block_idx)
            * globs.up_gate_proj_block_size
            * globs.hidden_size
            * 2  # gate and up
        )


@dataclass
class DownProjResidual(MatVecAdd):
    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return LayerNormDoubleMatVecSiLU.opcode()

    def cost(self, globs: Globals):
        return (
            (self.end_block_idx - self.start_block_idx)
            * globs.down_proj_block_size
            * globs.hidden_size
        )


@dataclass
class RMS_LM_Head(Instruction):
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()

    def cost(self, globs: Globals):
        return (
            (self.end_output_block_idx - self.start_output_block_idx)
            * globs.lm_head_block_size
            * globs.hidden_size
        )
