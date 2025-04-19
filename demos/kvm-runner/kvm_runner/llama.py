from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from kvm_runner.types import (
    BatchState,
    DeviceType,
    ExtraModelConfig,
)
from kvm_runner.utils import (
    load_safetensors_repo,
)

KV_Cache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        """
        Taken from LlamaRMSNorm.
        """
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


def all_gather(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra_config.tp_group)

    out = torch.empty(
        (extra_config.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.all_gather_into_tensor(out, x, group=extra_config.tp_group)
    return out


def reduce_scatter(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=extra_config.tp_group
        )

    out = torch.empty(
        (x.shape[0] // extra_config.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.reduce_scatter_tensor(out, x, group=extra_config.tp_group)
    return out


def attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    kv_cache: KV_Cache,
    position_ids: Tensor,
    seq_len: int,
) -> Tensor:
    num_new_tokens = query_states.shape[0]

    k_cache, v_cache = kv_cache

    k_cache[position_ids] = key_states
    v_cache[position_ids] = value_states

    def shape_for_sdpa(x: Tensor):
        return rearrange(x, "l h d -> h l d")

    def unshape_for_sdpa(x: Tensor):
        return rearrange(x, "h l d -> l h d")

    if num_new_tokens > 1:
        k_for_sdpa = shape_for_sdpa(key_states)
        v_for_sdpa = shape_for_sdpa(value_states)

        q_for_sdpa = shape_for_sdpa(query_states)

        # assume running prefill from scratch
        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=True, enable_gqa=True
        )
    else:
        # decode
        k_for_sdpa = shape_for_sdpa(k_cache[:seq_len])
        v_for_sdpa = shape_for_sdpa(v_cache[:seq_len])

        q_for_sdpa = shape_for_sdpa(query_states)

        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=False, enable_gqa=True
        )

    reshaped_attn_output = unshape_for_sdpa(attn_output)
    return reshaped_attn_output


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config)

        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim

        assert self.config.num_attention_heads % self.tp_size == 0
        assert (
            self.config.num_key_value_heads % self.tp_size == 0
            or self.config.num_key_value_heads == 1
        )

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim,
            config.hidden_size,
            bias=False,
        )

        self.kv_cache: KV_Cache | None = None

    def forward(
        self,
        batch_state: BatchState,
    ):
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert batch_state.position_ids is not None
        assert self.kv_cache is not None
        assert batch_state.seq_len is not None

        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)

        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, self.num_kv_heads, -1)

        cos, sin = batch_state.position_embeddings

        dtype = query_states.dtype

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,  # unsqueeze dim = head dim on q/k
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )

        attn_output = raw_attn_output.reshape(bsz, -1)

        o_proj = self.o_proj(attn_output)

        o_proj = reduce_scatter(o_proj, self.extra_config)

        with_residual = residual + o_proj

        batch_state.hidden_states = with_residual
        return batch_state


class LlamaMLP(nn.Module):
    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        assert self.config.intermediate_size % self.tp_size == 0
        self.intermediate_size = self.config.intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config)

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = self.input_layernorm(inp)

        hidden_states = all_gather(hidden_states, self.extra_config)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        with_residual = inp + down

        batch_state.hidden_states = with_residual
        return batch_state


class LlamaBlock(nn.Module):
    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(config, extra_config, layer_idx)
        self.mlp = LlamaMLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


class LlamaLMHead(nn.Module):
    def __init__(self, config: LlamaConfig, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config)

        self.tp_size = extra_config.tp_size or 1

        assert config.vocab_size % self.tp_size == 0
        head_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, head_size, bias=False)

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None

        hidden_states = batch_state.hidden_states

        if self.extra_config.tp_size > 1:
            hidden_states = all_gather(hidden_states, self.extra_config)

        hidden_states = self.input_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        next_token_ids = logits.argmax(dim=-1)

        if self.tp_size > 1:
            # TODO: fusion
            next_token_ids = all_gather(next_token_ids, self.extra_config)

        batch_state.output_ids = next_token_ids
        return batch_state


class LlamaEmbeddings(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)

        batch_state.hidden_states = hidden_states
        return batch_state


class LlamaModel(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: LlamaConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = LlamaEmbeddings(config)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(LlamaBlock(config, extra_config, i))

        self.layers = nn.ModuleList(layers)

        self.rope = LlamaRotaryEmbedding(
            config=config,
        )

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def forward(self, batch_state: BatchState):
        out: BatchState = self.embed_tokens(batch_state)
        assert self.rope_cos.dtype == torch.float32
        assert self.rope_sin.dtype == torch.float32
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)

        for layer in self.layers:
            out = layer(out)
        return out


@dataclass
class StackedParams:
    qkv_proj: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = LlamaModel(config, extra_config)

        self.lm_head = LlamaLMHead(config, extra_config)

    def num_kv_heads(self):
        all_heads = self.config.num_key_value_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def num_qo_heads(self):
        all_heads = self.config.num_attention_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def forward(
        self,
        batch_state: BatchState,
        async_tp: bool = False,
    ):
        self.async_tp = async_tp
        # making a copy of the input state - needed for cudagraphs + pp,
        # where we need to keep track of references to both the input
        # and output hidden states.
        out = BatchState(
            input_ids=batch_state.input_ids,
            position_ids=batch_state.position_ids,
            hidden_states=batch_state.hidden_states,
            seq_len=batch_state.seq_len,
        )

        out = self.model(out)
        out = self.lm_head(out)

        return out

    def setup_caches(self):
        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: LlamaBlock = self.model.layers[layer_idx]  # type: ignore
            layer.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config: LlamaConfig = LlamaConfig.from_pretrained(model_name_or_path)  # type: ignore
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            dtype = config.torch_dtype

        with init_empty_weights(include_buffers=False):
            model = cls(
                config,
                extra_config,
            )
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

            model_path = Path(snapshot_path_str)

        model.load_from_safetensors(model_path)

        # SE (10/18/24): It is important not to call model.to(device, dtype) because
        # this will convert the `inv_freq` buffer in the rotary embeddings to fp16
        # the HF load from pretrained is careful to not do this and keeps it in fp32.
        # The dtype for the parameters is already handled by the load calls above, but
        # it's possible that there are other buffers which *should* be converted to fp16.
        # TODO: it's probably easiest to figure out how we can just use HFs `load_from_pretrained`
        # to load the model weights so we can ensure that there are no other subtle differences
        model.to(device=device)

        model.requires_grad_(False)

        model.stack_params()
        model.setup_caches()

        return model

    def make_name_to_hf_name(self):
        keys = self.state_dict().keys()

        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[
                f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"
            ] = f"model.layers.{layer_idx}.input_layernorm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def make_tp_map(self):
        """
        Maps parameter names to the dimension they should be split on.
        Parameters that are not included in the map should not be split.
        """

        tp_map = {}
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0

            elif any(
                param_name.endswith(suffix)
                for suffix in ["o_proj.weight", "down_proj.weight"]
            ):
                tp_map[param_name] = 1

        return tp_map

    def load_from_safetensors(
        self,
        model_path: Path,
    ):
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}

        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        def stack_and_reassign(modules, prop: str):
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[LlamaBlock] = self.model.layers  # type: ignore
        self_attns = [x.self_attn for x in layers]
        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]

        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                ],
                dim=0,
            )

            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight

        self.stacked_params = StackedParams(
            qkv_proj=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
        )

    # def stack_params(self):
    #     state_dict = self.state_dict()
    #     name_to_stacked_param = defaultdict(lambda: {})

    #     state_dict_after_stacking = {}

    #     prefix = "model.layers."
    #     for name, param in state_dict.items():
    #         if name.startswith(prefix):
    #             after_prefix = name[len(prefix) :]
    #             layer_idx = int(after_prefix.split(".")[0])
    #             denumbered_name = name.replace(f"{layer_idx}", "all")
    #             name_to_stacked_param[denumbered_name][layer_idx] = param
    #         else:
    #             state_dict_after_stacking[name] = param

    #     stacked_params = {}

    #     for denumbered_name, number_to_param in name_to_stacked_param.items():
    #         num_layers = len(number_to_param)
    #         assert num_layers == self.config.num_hidden_layers
    #         to_stack = [number_to_param[i] for i in range(num_layers)]

    #         stacked = torch.stack(to_stack, dim=0)
    #         stacked_params[denumbered_name] = stacked

    #         for i in range(num_layers):
    #             og_name = denumbered_name.replace("all", f"{i}")
    #             state_dict_after_stacking[og_name] = stacked[i]

    #     self.load_state_dict(state_dict_after_stacking, assign=True, strict=True)
    #     self.stacked_params = stacked_params
