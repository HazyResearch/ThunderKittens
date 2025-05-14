from dataclasses import dataclass, fields

from kvm_runner.model_types import DeviceType
from kvm_runner.utils import get_sm_count
from torch import Tensor


@dataclass
class BaseGlobals:
    # model parameters, all layers stacked together in order
    qkv_proj_weights: Tensor
    attn_ln_weights: Tensor
    o_proj_weights: Tensor
    mlp_ln_weights: Tensor
    up_proj_weights: Tensor
    gate_proj_weights: Tensor
    down_proj_weights: Tensor
    lm_head_norm_weights: Tensor
    lm_head_weights: Tensor
    k_cache: Tensor
    v_cache: Tensor

    # not stacked for each layer
    rope_cos: Tensor
    rope_sin: Tensor

    # model constants
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    attn_scale: float
    rms_norm_eps: float
    device: DeviceType

    hidden_states: Tensor
    barriers: Tensor

    pos_id: int

    def __post_init__(self):
        self.instructions: Tensor | None = None
        self.timings: Tensor | None = None

    def sm_count(self) -> int:
        return get_sm_count(self.device)

    def diff(self, other: "BaseGlobals", skip_kv_cache: bool = True):
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)
            other_attr = getattr(other, name)
            if (
                not isinstance(attr, Tensor)
                or "weights" in name
                or name in ["rope_cos", "rope_sin"]
                or (skip_kv_cache and "cache" in name)
            ):
                continue
            diff_tensors(attr, other_attr, name)

    def num_total_heads(self) -> int:
        return self.num_attention_heads + self.num_kv_heads * 2


def diff_tensors(a: Tensor, b: Tensor, name: str):
    a = a.float()
    b = b.float()

    diff = a - b
    adiff = diff.abs()
    rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
    print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")
    return diff, adiff, rdiff


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
class PrintInfo:
    layer_filter: list[int] | None = None
    name_filter: list[str] | None = None
    state_filter: list[str] | None = None


@dataclass
class PrintState(Instruction):
    layer_idx: int
    name: str
    print_info: PrintInfo
