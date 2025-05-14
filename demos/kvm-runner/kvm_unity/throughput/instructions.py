from dataclasses import dataclass

from kvm_unity.instructions import BaseGlobals, Instruction
from kvm_unity.utils import assert_div
from torch import Tensor


@dataclass
class Globals(BaseGlobals):
    # activation buffers
    hidden_states: Tensor
    rms_rope_intermediates: Tensor
    rms_gate_intermediates: Tensor
    rms_lm_head_intermediates: Tensor

    post_ln_rope_q: Tensor
    attn_out: Tensor
    silu_out: Tensor
    logits: Tensor

    batch_size: int

    matmul_batch_block_size: int
    matmul_output_block_size: int
    norm_block_size: int

    def num_batch_blocks(self) -> int:
        return assert_div(self.batch_size, self.matmul_batch_block_size)

    def num_output_blocks(self) -> int:
        return assert_div(self.hidden_size, self.matmul_output_block_size)


@dataclass
class PreAttnLayerNorm(Instruction):
    """
    pre-attention layernorm
    """

    layer_idx: int
    batch_start_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()


@dataclass
class QKV_MatMulRopeAppend(Instruction):
    """
    attention: qkv matmul + rope on q and k + append k and v to kv cache
    """

    layer_idx: int
    batch_start_idx: int
    qkv_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return PreAttnLayerNorm.opcode()


@dataclass
class AttentionDecode(Instruction):
    layer_idx: int
    batch_start_idx: int
    kv_head_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return QKV_MatMulRopeAppend.opcode()


@dataclass
class MatMulAdd(Instruction):
    layer_idx: int
    batch_start_idx: int
    output_block_idx: int


# denoting these with separate opcodes so that know what inputs to read from


@dataclass
class O_ProjResidual(MatMulAdd):
    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> int:
        return AttentionDecode.opcode()


@dataclass
class PreMLP_Norm(Instruction):
    """
    layernorm pre-mlp
    """

    layer_idx: int
    batch_start_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return O_ProjResidual.opcode()


@dataclass
class GateSilu(Instruction):
    """
    matmul + silu
    """

    layer_idx: int
    batch_start_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return PreMLP_Norm.opcode()


@dataclass
class UpMatMul(Instruction):
    """
    matmul + gate
    """

    layer_idx: int
    batch_start_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return GateSilu.opcode()


@dataclass
class DownProjResidual(MatMulAdd):
    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> int:
        return UpMatMul.opcode()


@dataclass
class PreLMHeadRMS(Instruction):
    batch_start_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()


@dataclass
class LM_Head(Instruction):
    batch_start_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> int:
        return PreLMHeadRMS.opcode()
