from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

KV_Cache = tuple[Tensor, Tensor]
DeviceType = torch.device | str


@dataclass
class ModelOutput:
    schedule_id: str
    output_tokens: list[int]
    logprobs: list[float]
    microbatch_index: int | None = None


@dataclass
class BatchState:
    input_ids: Tensor
    position_ids: Tensor | None = None
    seq_len: int | None = None
    output_ids: Tensor | None = None
    hidden_states: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None

    def __post_init__(self):
        if self.seq_len is None:
            self.seq_len = self.input_ids.numel()


@dataclass
class ExtraModelConfig:
    """
    For flags that we define that aren't in the hf config.
    """

    tp_size: int = 1
    tp_rank: int = 0
    tp_group: dist.ProcessGroup | None = None

    torch_compile: bool = False

    rope_scaling: dict | None = None

    interleave_rope: bool = False
