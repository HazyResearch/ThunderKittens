from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers import LlamaConfig
from torch import nn
from typing import Optional, Tuple, Union, List
import torch

from typing import Optional, Tuple, Union, List

from tk_hedgehog_window_attention import TKHedgehogWindowAttention


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = TKHedgehogWindowAttention(config=config, layer_idx=layer_idx)


class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
