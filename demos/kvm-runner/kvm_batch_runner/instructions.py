from dataclasses import dataclass, fields
from typing import Optional

from kvm_batch_runner.model_types import DeviceType
from kvm_batch_runner.utils import get_sm_count
from torch import Tensor


@dataclass
class Globals:
    # model parameters, all layers stacked together in order
    qkv_proj: Tensor
    attn_ln_weight: Tensor
    o_proj: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor
    lm_head_norm_weights: Tensor
    lm_head_weights: Tensor
    k_cache: Tensor
    v_cache: Tensor

    # not stacked for each layer
    rope_cos: Tensor
    rope_sin: Tensor

    # activation buffers
    hidden_states: Tensor
    post_ln_rope_q: Tensor
    attn_out: Tensor
    silu_out: Tensor
    logits: Tensor

    pos_id: int
    attn_scale: float
    rms_norm_eps: float

    # block size constants
    matmul_silu_block_size: int
    matmul_gate_block_size: int
    down_proj_block_size: int
    o_proj_block_size: int
    lm_head_block_size: int
    matmul_block_size: int
    qkv_block_size: int
    attn_kv_block_size: int

    batch_block_size: int
    # attn_reduction_size: int

    # model constants
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    device: DeviceType

    # the vm stuff (instructions and timing) - sizes dependent on the generated instructions
    instructions: Tensor | None = None
    timings: Tensor | None = None
    barriers: Tensor | None = None

    def sm_count(self) -> int:
        return get_sm_count(self.device)


@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def prev_opcode(cls) -> int:
        raise NotImplementedError

    def serialize(self):
        words = [self.opcode()]
        for field in fields(self):
            name = field.name
            if name == "global_idx":
                continue
            attr = getattr(self, name)

            if isinstance(attr, int):
                words.append(attr)
            elif isinstance(attr, tuple):
                words.append(len(attr))
                words.extend(attr)
            elif isinstance(attr, list):
                words.append(len(attr))
                words.extend(attr)
            # for convenience
            elif attr is None:
                words.append(0)
            else:
                raise ValueError(f"Unsupported field type: {attr}")

        return words


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0


@dataclass
class PreLayerNorm(Instruction):
    """
    pre-attention layernorm
    """

    layer_idx: int
    batch_start_idx: int 
    batch_end_idx: int

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
    batch_end_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return PreLayerNorm.opcode()


@dataclass
class AttentionDecode(Instruction):
    layer_idx: int
    batch_start_idx: int
    batch_end_idx: int
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
    batch_end_idx: int
    output_block_idx: int
    reduction_block_idx: int  # in units of 2048


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
class PostLayerNorm(Instruction):
    """
    layernorm post-attention
    """

    layer_idx: int
    batch_start_idx: int 
    batch_end_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return O_ProjResidual.opcode()


@dataclass
class MatMulSiLU(Instruction):
    """
    matmul + silu
    """

    layer_idx: int
    batch_start_idx: int 
    batch_end_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return PostLayerNorm.opcode()


@dataclass
class MatMulGate(Instruction):
    """
    matmul + gate
    """

    layer_idx: int
    batch_start_idx: int 
    batch_end_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return MatMulSiLU.opcode()


@dataclass
class DownProjResidual(MatMulAdd):
    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> int:
        return MatMulGate.opcode()


@dataclass
class RMS_LM_Head(Instruction):
    batch_start_idx: int
    batch_end_idx: int
    output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()


@dataclass
class PrintInfo:
    layer_filter: list[int] | None = None
    name_filter: list[str] | None = None
    state_filter: list[str] | None = None


@dataclass
class PrintState(Instruction):
    layer_idx: int
    name: str
    print_info: PrintInfo
