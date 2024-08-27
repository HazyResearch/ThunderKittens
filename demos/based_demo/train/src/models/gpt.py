# Copyright (c) 2023, Tri Dao.

import logging
import math
import re
from collections import namedtuple
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
import hydra

from .block import Block, ParallelBlock
from .embeddings import GPT2Embeddings
from .mha import MHA
from .mlp import (
    FusedMLP,
    Mlp,
)
from .mlp import GatedMlp, ParallelGatedMlp
try:
    from flash_attn.ops.activations import sqrelu_fwd
except:
    sqrelu_fwd = None
try:
    from flash_attn.utils.distributed import all_gather_raw, sync_shared_params
except:
    all_gather_raw, sync_shared_params = None, None
from train.src.generation import GenerationMixin, NaiveGenerationMixin
# from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None
    from based.ops.layer_norm import RMSNorm
    print("Using triton RMSNorm; may see slight output differences.")

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None

logger = logging.getLogger(__name__)

# torch.backends.cuda.matmul.allow_tf32 = False   # FLAG
from train.src.utils.hf import load_config_hf, load_state_dict_hf


class GPT2MixerConfig(GPT2Config):
    def __init__(self, *args, **kwargs):
        self.mixer = kwargs.pop("mixer", None)
        super().__init__(*args, **kwargs)


def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    tag = 'mixer'
    value = getattr(config, "mixer", None)
    alt_mixer_layers = getattr(config, "alt_mixer_layers", None)
    alt_mixer_2_layers = getattr(config, "alt_mixer_2_layers", None)
    alt_mixer = getattr(config, "alt_mixer", None)
    alt_mixer_2 = getattr(config, "alt_mixer_2", None)
    if alt_mixer_2_layers is not None and layer_idx in alt_mixer_2_layers:
        value = None
        if alt_mixer_2 is not None:
            tag = 'alt_mixer_2'
            value = config.alt_mixer_2
    elif alt_mixer_layers is not None and layer_idx in alt_mixer_layers:
        value = None
        if alt_mixer is not None:
            tag = 'alt_mixer'
            value = config.alt_mixer
    if (not hasattr(config, tag)) or value is None:
        return create_mha_cls(config, layer_idx, process_group=process_group, device=device)
    import os
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # SA: fork whether to use the FA settings to initialize the mixer (e.g. for rotary_emb_dim)
    # Leave as fork for compatibility with prior models we've trained.
    wds = ['slide', 'sliding', 'window']
    if any(wd in value['_target_'] for wd in wds): # and 'do_update' in value:
        value = sliding_window_additions(value, config, layer_idx=layer_idx, process_group=process_group, device=device)

    mixer_cls = hydra.utils.instantiate(
        value, 
        _partial_=True, 
        device=device, 
        dtype=dtype, 
        layer_idx=layer_idx,
    )
    return mixer_cls


def sliding_window_additions(value, config, layer_idx, process_group, device):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)

    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)

    serial_kwargs = (
        {"fused_bias_fc": fused_bias_fc} if process_group is None else {}
    )
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", True),
        }
        if process_group is not None
        else {}
    )
    num_heads_kv = getattr(config, "num_head_kv", None)
    if num_heads_kv is None:
        num_heads_kv = getattr(config, "n_head_kv", None)

    value.update({
        "num_heads": config.num_attention_heads,
        "num_heads_kv": num_heads_kv,
        "qkv_proj_bias": qkv_proj_bias,
        "out_proj_bias": out_proj_bias,
        "dropout": config.attn_pdrop,
        "softmax_scale": softmax_scale,
        "causal": True,
        "layer_idx": layer_idx,
        "rotary_emb_dim": rotary_emb_dim,
        "rotary_emb_base": rotary_emb_base,
        "rotary_emb_scale_base": rotary_emb_scale_base,
        "rotary_emb_interleaved": rotary_emb_interleaved,
        "use_flash_attn": use_flash_attn,
        **serial_kwargs,
        **parallel_kwargs,
    })
    return value


def create_mha_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    if dwconv:
        assert process_group is None, "TensorParallel MHA does not support dwconv yet"
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    if not fused_bias_fc:
        assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = (
        {"fused_bias_fc": fused_bias_fc, "dwconv": dwconv} if process_group is None else {}
    )
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", True),
        }
        if process_group is not None
        else {}
    )
    num_heads_kv = getattr(config, "num_head_kv", None)
    if num_heads_kv is None:
        num_heads_kv = getattr(config, "n_head_kv", None)
    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        num_heads_kv=num_heads_kv,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_flash_attn=use_flash_attn,
        **serial_kwargs,
        **parallel_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None, return_residual=False):
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)
    fused_mlp = getattr(config, "fused_mlp", False)

    if fused_mlp:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
        ]
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only " "supports approximate activation_function sqrelu"
        )
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if not fused_mlp and not fused_dense_sqrelu_dense:
        # print(f"mlp inner_dim -- not fused: {config.n_inner}")
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            activation = (
                F.sigmoid
                if config.activation_function == "glu"
                else (F.silu if config.activation_function == "swiglu" else F.gelu)
            )
            mlp_cls = GatedMlp if process_group is None else ParallelGatedMlp
            # print(f"{mlp_cls=}")
            parallel_kwargs = (
                {
                    "process_group": process_group,
                    "sequence_parallel": getattr(config, "sequence_parallel", True),
                }
                if process_group is not None
                else {}
            )
            mlp_type = getattr(config, "mlp_type", 'base')
            mlp_cls = partial(
                mlp_cls,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                mlp_type=mlp_type,
                ff_mult=getattr(config, "ff_mult", 2),
                return_residual=return_residual,
                **parallel_kwargs,
                **factory_kwargs,
            )
        else:
            if config.activation_function == "relu":
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == "sqrelu" and sqrelu_fwd is not None:
                activation = sqrelu_fwd
            else:
                approximate = (
                    "tanh"
                    if config.activation_function
                    in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]
                    else "none"
                )
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = Mlp if process_group is None else ParallelMLP
            parallel_kwargs = (
                {
                    "process_group": process_group,
                    "sequence_parallel": getattr(config, "sequence_parallel", True),
                }
                if process_group is not None
                else {}
            )
            mlp_cls = partial(
                mlp_cls,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                return_residual=return_residual,
                **parallel_kwargs,
                **factory_kwargs,
            )
    else:
        print(f"mlp inner_dim -- fused: {config.n_inner}")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError("fused_dense is not installed")
            activation = (
                "gelu_approx"
                if config.activation_function
                in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]
                else config.activation_function
            )
            mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
            parallel_kwargs = (
                {
                    "process_group": process_group,
                    "sequence_parallel": getattr(config, "sequence_parallel", True),
                }
                if process_group is not None
                else {}
            )
            mlp_cls = partial(
                mlp_cls,
                hidden_features=config.n_inner,
                activation=activation,
                checkpoint_lvl=mlp_checkpoint_lvl,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                return_residual=return_residual,
                **parallel_kwargs,
                **factory_kwargs,
            )
        elif fused_dense_sqrelu_dense:
            if process_group is not None:
                assert fused_mlp, "Tensor Parallel is not implemented for FusedDenseSqreluDense"
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(
                FusedDenseSqreluDense,
                return_residual=return_residual,
                hidden_features=config.n_inner,
                checkpoint_lvl=mlp_checkpoint_lvl,
                **factory_kwargs,
            )
        else:
            raise RuntimeError("MLP type not supported")
    return mlp_cls


def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None, **kwargs):
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    use_rms_norm = getattr(config, "rms_norm", False)
    
    try:
        norm_cls = partial(
            nn.LayerNorm if not use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
            **factory_kwargs,
        )
    except:
        print("please install the fused layernorm kernel as specified in the README")
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    parallel_block = getattr(config, "parallel_block", False)
    if not parallel_block:
        block = Block(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            prenorm=prenorm,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
            layer_idx=layer_idx,
        )
    else:
        assert prenorm
        block = ParallelBlock(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            tied_norm=getattr(config, "parallel_block_tied_norm", False),
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
        )
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        config,
        *args,
        strict=True,
        device=None,
        dtype=None,
        world_size=1,
        rank=0,
        **kwargs,
    ):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model
        
    @classmethod
    def from_pretrained_hf(cls, pretrained_model_name, device=None, implementation='default', recurrent_impl="default", inference_bs=1, swa_inference_mode="default", override_seqlen=None, override_model_dims=None, silent=True, **kwargs):

        config_data = load_config_hf(pretrained_model_name)
        config = GPT2Config(**config_data)

        try:
            config.alt_mixer['inference_implementation'] = implementation
            config.alt_mixer['recurrent_impl'] = recurrent_impl
            config.alt_mixer['inference_bs'] = inference_bs
            config.alt_mixer['silent'] = silent
        except:
            pass

        try:
            config.alt_mixer_2['inference_bs'] = inference_bs
            config.alt_mixer_2['l_max'] = config.mixer['l_max'] 
            config.alt_mixer_2['inference_mode'] = swa_inference_mode
        except:
            pass


        # These two are for benchmarking purposes
        if override_seqlen is not None:
            # for benchmarking
            try:
                config.alt_mixer['l_max'] = override_seqlen  
                config.mixer['l_max'] = override_seqlen 
                config.alt_mixer_2['l_max'] =override_seqlen
            except:
                pass

        model = cls(config, device=device, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{num_params=:.2e}")

        # remove the 'model.' prefix from the keys
        state_dict = {re.sub("^model\.", "", k): v for k, v in state_dict.items()}
        # remove Unexpected key(s) in state_dict: "train_metrics.num-tokens.count", "val_metrics.num-tokens.count", "test_metrics.num-tokens.count". from the state_dict
        state_dict = {k: v for k, v in state_dict.items() if "metrics" not in k}

        if override_seqlen is None and override_model_dims is None: 
            model.load_state_dict(state_dict)
        return model.to(device=device)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, use_weight_init=True):
    if isinstance(module, nn.Linear):
        if use_weight_init:
            nn.init.normal_(module.weight, std=initializer_range)   # SA: this line isn't in Mamba init code
        else:
            print(f"Skipping!")
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class DecayClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l_max = config.mixer.get('l_max', None)
        if self.l_max is None:
            self.l_max = config.alt_mixer.get('l_max', None)
        assert self.l_max > 0, print(f'double check l_max')
        decay_const = getattr(config, "decay_const", -3)
        decay = torch.log(1 - 2 ** (decay_const - torch.arange(config.n_head, dtype=torch.float)))
        self.register_buffer("decay", decay)
    
    def forward(self):
        index = torch.arange(self.l_max).to(self.decay)
        mask = torch.tril(torch.ones(self.l_max, self.l_max).to(self.decay))
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp(mask * self.decay[:, None, None])
        mask = torch.nan_to_num(mask)
        return mask, torch.exp(self.decay)


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.process_group = process_group
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        # For GPT-J, GPT-NeoX
        self.parallel_block = getattr(config, "parallel_block", False)

        self.enc_length = getattr(config, "enc_length", None)

        self.embeddings = GPT2Embeddings(
            config.hidden_size,
            vocab_size,
            config.max_position_embeddings,
            word_embed_proj_dim=word_embed_proj_dim,
            **factory_kwargs,
        )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(config, layer_idx=i, process_group=process_group, **factory_kwargs)
                for i in range(config.num_hidden_layers)
            ]
        )
        if dropout_add_layer_norm is None:
            self.fused_dropout_add_ln = False 
            print("Please install fused_dense following the README for additional speedup.")
        else:
            self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if (not self.parallel_block and dropout_add_layer_norm is None) or (
                self.parallel_block and dropout_add_layer_norm_parallel_residual is None
            ):
                raise ImportError("dropout_layer_norm is not installed")
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(
                config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs
            )
        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True
        
        if getattr(config, "special_initializer", False):
            initializer_range = (2 / (config.n_embd * 5)) ** 0.5
        else:
            initializer_range = config.initializer_range

        if getattr(config, 'fixed_decay', False):
            self.decay = DecayClass(config)
        else:
            self.decay = None

        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                use_weight_init=getattr(config, "use_weight_init", True),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None, attention_mask=None, stream=None, **kwargs):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention layers need to know the seqlen.
        if 1:
            embedding_kwargs = (
                {"combine_batch_seqlen_dim": True}
                if self.process_group is not None and self.sequence_parallel
                else {}
            )
            hidden_states = self.embeddings(
                input_ids, 
                position_ids=position_ids, 
                **embedding_kwargs
            )
            if self.parallel_block:
                hidden_states2 = None
            residual = None
            mixer_kwargs = (
                {"seqlen": input_ids.shape[1]}
                if self.process_group is not None and self.sequence_parallel
                else {}
            )
            if 'mask' in kwargs['kwargs']:
                mixer_kwargs['mask'] = kwargs['kwargs']['mask']
            if 'attn_mask' in kwargs['kwargs']:
                mixer_kwargs['attn_mask'] = kwargs['kwargs']['attn_mask']
            if inference_params is not None:
                mixer_kwargs["inference_params"] = inference_params
                mixer_kwargs['stream'] = stream

            # decay
            if self.decay is not None:
                decay = self.decay()
            else:
                decay = None

            for layer in self.layers:
                if self.prenorm:
                    layer_name = layer.mixer.__class__.__name__
                    if not self.parallel_block and layer_name not in ['MHA']:
                        hidden_states, residual = layer(
                            hidden_states, residual=residual, position_ids=position_ids, decay=decay, mixer_kwargs=mixer_kwargs
                        )
                    elif not self.parallel_block and layer_name in ['MHA']:
                        hidden_states, residual = layer(hidden_states, residual=residual, mixer_kwargs=mixer_kwargs)
                    else:
                        hidden_states, hidden_states2, residual = layer(
                            hidden_states, hidden_states2, residual=residual, position_ids=position_ids, decay=decay, mixer_kwargs=mixer_kwargs
                        )
                else:
                    hidden_states = layer(hidden_states, position_ids=position_ids, mixer_kwargs=mixer_kwargs)

            if self.prenorm:
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_f(hidden_states)
                    if not self.parallel_block:
                        residual = (dropped + residual) if residual is not None else dropped
                    else:
                        dropped2 = self.drop_f(hidden_states2)
                        residual = (
                            (residual + dropped + dropped2)
                            if residual is not None
                            else dropped + dropped2
                        )
                    hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    if not self.parallel_block:
                        fused_add_norm_fn = (
                            dropout_add_rms_norm
                            if isinstance(self.ln_f, RMSNorm)
                            else dropout_add_layer_norm
                        )
                        hidden_states = fused_add_norm_fn(
                            hidden_states,
                            residual,
                            self.ln_f.weight,
                            self.ln_f.bias,
                            self.drop_f.p if self.training else 0.0,
                            self.ln_f.eps,
                            prenorm=False,
                            residual_in_fp32=self.residual_in_fp32,
                        )
                    else:
                        fused_add_norm_fn = (
                            dropout_add_rms_norm_parallel_residual
                            if isinstance(self.ln_f, RMSNorm)
                            else dropout_add_layer_norm_parallel_residual
                        )
                        hidden_states, _ = fused_add_norm_fn(
                            hidden_states,
                            hidden_states2,
                            residual,
                            self.ln_f.weight,
                            self.ln_f.bias,
                            None,
                            None,
                            self.drop_f.p if self.training else 0.0,
                            self.ln_f.eps,
                            prenorm=False,
                            residual_in_fp32=self.residual_in_fp32,
                        )
            return hidden_states


class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin, NaiveGenerationMixin):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.process_group = process_group
        self.transformer = GPTModel(config, process_group=process_group, **factory_kwargs)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None
        if process_group is None:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                embed_dim,
                vocab_size,
                process_group,
                bias=lm_head_bias,
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
            )
        )
        self.tie_weights()
       
    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, stream=None, **kwargs):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        if type(input_ids) == list:
            input_ids = input_ids[0]    
        assert (
            input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params,
            stream=stream, kwargs=kwargs
        )
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if ColumnParallelLinear is not None and isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, "(n b) ... d -> b ... (n d)", b=b)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.weight")
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f"transformer.layers.{l - 1}.norm2.weight")
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict[f"transformer.layers.0.norm1.weight"] = ln_weight
            state_dict[f"transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)

