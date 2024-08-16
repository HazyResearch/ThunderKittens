# Copyright (c) 2023, Tri Dao.
# Adapted by Simran Arora and Sabri Eyuboglu.

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_func,
)

import inspect
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
assert _flash_supports_window_size, "flash_attn_func does not support window_size"

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

from based.models.mixers.mixer_utils.rotary import get_rotary_embeddings, apply_rotary_pos_emb

class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=True, softmax_scale=None, attention_dropout=0.0, window_size=None):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.window_size = window_size

    def forward(self, qkv, causal=True, cu_seqlens=None, max_seqlen=None, **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        in_dtype = qkv.dtype
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.bfloat16)

        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = True # Changed (01152023)
        unpadded = cu_seqlens is not None

        if self.window_size is not None:
            if unpadded:
                assert cu_seqlens.dtype == torch.int32
                assert max_seqlen is not None
                assert isinstance(max_seqlen, int)
                return flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                    window_size=self.window_size ,    # [i - window_size[0], i + window_size[1]]
                ).to(in_dtype)
            else:
                return flash_attn_qkvpacked_func(
                    qkv,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                    window_size=self.window_size,  
                ).to(in_dtype)
        else:
            assert 0, print("Using windows")
            if unpadded:
                assert cu_seqlens.dtype == torch.int32
                assert max_seqlen is not None
                assert isinstance(max_seqlen, int)
                return flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                ).to(in_dtype)
            else:
                return flash_attn_qkvpacked_func(
                    qkv,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                ).to(in_dtype)


class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=True, softmax_scale=None, attention_dropout=0.0, window_size=None):
        super().__init__()
        assert flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.window_size = window_size

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        in_dtype = q.dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            q = q.to(torch.bfloat16)
            kv = kv.to(torch.bfloat16)
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = True #self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.window_size is not None:
            if unpadded:
                assert cu_seqlens.dtype == torch.int32
                assert max_seqlen is not None
                assert isinstance(max_seqlen, int)
                assert cu_seqlens_k is not None
                assert cu_seqlens_k.dtype == torch.int32
                assert max_seqlen_k is not None
                assert isinstance(max_seqlen, int)
                return flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens_k,
                    max_seqlen,
                    max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    window_size=self.window_size,
                ).to(in_dtype)
            else:
                batch_size, seqlen_q = q.shape[0], q.shape[1]
                seqlen_k = kv.shape[1]
                assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
                return flash_attn_kvpacked_func(
                    q,
                    kv.to(q.dtype),
                    self.drop.p if self.training else 0.0,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                    window_size=self.window_size,
                ).to(in_dtype)
        else:
            assert 0, print("Using windows")
            if unpadded:
                assert cu_seqlens.dtype == torch.int32
                assert max_seqlen is not None
                assert isinstance(max_seqlen, int)
                assert cu_seqlens_k is not None
                assert cu_seqlens_k.dtype == torch.int32
                assert max_seqlen_k is not None
                assert isinstance(max_seqlen, int)
                return flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens_k,
                    max_seqlen,
                    max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                ).to(in_dtype)
            else:
                batch_size, seqlen_q = q.shape[0], q.shape[1]
                seqlen_k = kv.shape[1]
                assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
                return flash_attn_kvpacked_func(
                    q,
                    kv,
                    self.drop.p if self.training else 0.0,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                ).to(in_dtype)


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx, inference_mode='default', window_size=None):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""

    if inference_mode == "default" or inference_mode == "default_rotary":
        # Pre-allocate memory for key-values for inference.
        num_heads, head_dim = kv.shape[-2:]
        if layer_idx not in inference_params.key_value_memory_dict:
            kv_cache = torch.empty(
                inference_params.max_batch_size,
                inference_params.max_seqlen,
                2,
                num_heads,
                head_dim,
                dtype=kv.dtype,
                device=kv.device,
            )
            inference_params.key_value_memory_dict[layer_idx] = kv_cache
        else:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        # Adjust key and value for inference
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + kv.shape[0]
        sequence_start = inference_params.seqlen_offset
        sequence_end = sequence_start + kv.shape[1]
        assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else kv_cache.shape[0])
        assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else kv_cache.shape[2])
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv

        return kv_cache[batch_start:batch_end, :sequence_end, ...], None, None
    else:
        assert window_size is not None, print("Implementing this for SWA.")

        cache_size = min(inference_params.max_seqlen, window_size) 
        # Pre-allocate memory for key-values for inference.
        num_heads, head_dim = kv.shape[-2:]
        if layer_idx not in inference_params.key_value_memory_dict:
            kv_cache = torch.empty(
                inference_params.max_batch_size,
                cache_size,    # SA: Change 1
                2,
                num_heads,
                head_dim,
                dtype=kv.dtype,
                device=kv.device,
            )
            inference_params.key_value_memory_dict[layer_idx] = kv_cache
        else:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]

        # Adjust key and value for inference
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + kv.shape[0]
        sequence_start = inference_params.seqlen_offset
        sequence_end = sequence_start + kv.shape[1]
        
        assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else kv_cache.shape[0])
        # assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else kv_cache.shape[2]) # SA: Change 2
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, :min(kv.shape[1], cache_size), ...] = kv[:, max(0,(kv.shape[1]-cache_size)):,:,:,:] # SA: Change 3]
        return kv_cache[batch_start:batch_end, :sequence_end, ...], cache_size, kv


class SlidingAttention(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=True,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        fused_bias_fc=False,
        use_flash_attn=True,
        return_residual=False,
        checkpointing=False,
        window_size=128,
        l_max=None,
        inference_bs=1,
        inference_mode="default",
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        # SA hardcode
        use_flash_attn = True
        causal = True

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        if window_size is None:
            assert 0, print("Using windows")
            self.window = None
        else:
            self.window = (window_size//2, 0)
        self.window_size = self.window[0]

        # inference settings
        self.inference_bs = inference_bs
        self.inference_mode= inference_mode# "default_rotary" # SA: Flag 2
        self.cache_size = self.window_size + 1

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (self.num_heads % self.num_heads_kv == 0), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            if self.inference_mode == "default":
                self.rotary_emb = RotaryEmbedding(
                    self.rotary_emb_dim,
                    base=rotary_emb_base,
                    scale_base=rotary_emb_scale_base,
                    interleaved=rotary_emb_interleaved,
                    device=device,
                )
            else:
                position_ids = torch.arange(l_max).to(device)
                self.position_ids = position_ids.unsqueeze(0).repeat(inference_bs, 1)
                # Rotary embeddings
                self.rotary_emb = get_rotary_embeddings(
                    rope_scaling_type=None,
                    head_dim=self.rotary_emb_dim,
                    max_position_embeddings=l_max,
                    rope_theta=rotary_emb_base,
                    rope_scaling_factor=rotary_emb_scale_base, 
                    device=device,
                )

        if fused_bias_fc and FusedDense is None: raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        )
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = FlashSelfAttention
        inner_cross_attn_cls = FlashCrossAttention 
        self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, window_size=self.window
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, window_size=self.window
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if self.inference_mode == "default" or self.inference_mode == "default_rotary": 
            return torch.empty(batch_size,max_seqlen,2,self.num_heads_kv,self.head_dim,dtype=dtype,device=device,)
        else:
            return torch.empty(batch_size,self.window_size,2,self.num_heads_kv,self.head_dim,dtype=dtype,device=device,)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        kv_cache, cache_size, cur_kv =  _update_kv_cache(kv, inference_params, self.layer_idx, inference_mode = self.inference_mode, 
        window_size=self.window_size)
        self.cache_size = cache_size
        return kv_cache, cur_kv

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn, "This code path only supports FlashAttention"
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        in_dtype = q.dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            q = q.to(torch.bfloat16)
            kv_cache = kv_cache.to(torch.bfloat16)
            kv = kv.to(torch.bfloat16)

        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None

        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )

        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0].to(q.dtype),
            kv_cache[:, :, 1].to(q.dtype),
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            window_size=self.window,
        ).to(in_dtype)
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv_cache, cur_kv = self._update_kv_cache(kv, inference_params)
            if self.inference_mode == "default" or self.inference_mode == "default_rotary": 
                return self.inner_cross_attn(q, kv_cache)
            else:
                return self.inner_cross_attn(q, cur_kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                # inference_params.lengths_per_sample[:batch]
                # if inference_params.lengths_per_sample is not None
                # else 
                inference_params.seqlen_offset
            )
            # breakpoint()
            in_dtype = q.dtype
            if q.dtype not in [torch.float16, torch.bfloat16]:
                q = q.to(torch.float16)
                kv_cache = kv_cache.to(torch.float16)
                kv = kv.to(torch.float16)
            if self.window is None:
                assert 0, print("Using windows")
                return flash_attn_with_kvcache(
                    q,
                    kv_cache[:, :, 0],
                    kv_cache[:, :, 1],
                    kv[:, :, 0],
                    kv[:, :, 1],
                    cache_seqlens=cache_seqlens,
                    softmax_scale=self.inner_cross_attn.softmax_scale,
                    causal=self.inner_cross_attn.causal,
                ).to(in_dtype)
            else:
                
                if self.inference_mode == "default" or self.inference_mode == "default_rotary" or cache_seqlens < self.cache_size:
                    length = cache_seqlens
                else:
                    length = self.cache_size-1

                # circular buffering
                if self.cache_size and self.cache_size < cache_seqlens:
                    kv_cache = torch.roll(kv_cache, -1, dims=1) # make room at the end
                    kv_cache[:,-1,:,:,:] = torch.zeros_like(kv_cache[:,-1,:,:,:])   # clear the end

                out = flash_attn_with_kvcache(
                    q,
                    kv_cache[:, :, 0],
                    kv_cache[:, :, 1],
                    kv[:, :, 0],
                    kv[:, :, 1],
                    cache_seqlens=length,
                    softmax_scale=self.inner_cross_attn.softmax_scale,
                    causal=self.inner_cross_attn.causal,
                    window_size=self.window,
                ).to(in_dtype)
                return out

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """

        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual: qkv = self.Wqkv(x)
            else: qkv, x = self.Wqkv(x)
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn or 
                self.inference_mode != "default"
            ):
                if self.rotary_emb_dim > 0: 
                    if self.inference_mode == "default":

                        qkv = self.rotary_emb(
                            qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                        )
                    else:
                        cos, sin = self.rotary_emb(qkv, seq_len=rotary_max_seqlen)  
                        q, k = apply_rotary_pos_emb(
                            qkv[:,:,0,:,:].transpose(1,2), 
                            qkv[:,:,1,:,:].transpose(1,2), 
                            cos, sin, self.position_ids[:,:seqlen]
                        )
                        qkv = torch.concat([q.transpose(1,2).unsqueeze(2),k.transpose(1,2).unsqueeze(2), qkv[:,:,2,:,:].unsqueeze(2),], dim=2)

                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
            else:
                context = self._apply_rotary_update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            assert self.num_heads_kv != self.num_heads
            if not self.return_residual: qkv = self.Wqkv(x)
            else: qkv, x = self.Wqkv(x)
            q = qkv[..., : self.num_heads * self.head_dim]
            kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

