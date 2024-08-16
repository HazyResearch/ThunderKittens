from typing import Optional, Tuple, List
import math
import os 
import sys
import copy

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention, Cache, 
    apply_rotary_pos_emb, repeat_kv
)

from thunderkittens import hedgehog as tk_window_hedgehog_attention

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, 'build/lib.linux-x86_64-cpython-312'))
# import h100_fwd as mod

from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_pad_to_multiple(n: int, m: int = 64) -> int:
    """Return padding to get n to nearest greatest multiple of m"""
    return m - (n % m)


class HedgehogFeatureMap(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, feature_dim: int, 
                 dtype: torch.dtype, device: torch.device, eps: float=1e-12):
        super().__init__()
        self.layer = nn.Parameter(torch.zeros(
            (num_heads, head_dim, feature_dim),
            dtype=dtype, device=device,
        ))
        nn.init.kaiming_uniform_(self.layer)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, num_heads, seq_len, head_dim)
        """
        x = torch.einsum('hdf,bhld->bhlf', self.layer, x)
        return torch.cat([
            torch.softmax(x, dim=-1), torch.softmax(x, dim=-1)
        ], dim=-1).clamp(self.eps)



class TKHedgehogWindowAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.window_size = 64   # hard-coded support for TK kernel
        self.layer_idx = layer_idx
        assert self.layer_idx is not None
        device = self.q_proj.weight.device
        dtype  = self.q_proj.weight.dtype
        self.register_buffer(  # dummy weights for combining sliding window + linear attn terms
            'window_factors', 1 * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
        )
        self.affine_attention_factors = False  # also how we combine; dummy value
        self.feature_map_q = HedgehogFeatureMap(self.num_heads, self.head_dim, 
                                                feature_dim=self.head_dim // 2,
                                                dtype=dtype, device=device)
        self.feature_map_k = copy.deepcopy(self.feature_map_q)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        TK Hedgehog linear attention + sliding window attention

        For prefill: 
        - We compute the sliding window of softmax attentions in a "terraced" manner 
        - We compute the remaining terms as linear attentions
        - We use the TK kernel to handle this entirely

        For generating: (see TKHedgehogWindowAttentionState)  
        - The TK kernel first gives us a KV state and K state that are computed from 
          token positions 1-64 tokens before the final query we process
          - This depends on the amount of padding we need to do to get the prefill
            to a multiple of 64. 
            - If there's none needed, the KV and K state are 64 tokens back. 
            - Otherwise they are (64 - padding) tokens back
          - This in-turn influences how many K and V we store in the KV cache. Where we 
            directly save (64 - padding) K and V into the KV cache
        - Then for subsequent generations, if len(KV cache) < 64 tokens, we just fill up the cache
          - Otherwise, we evict the first K and V from the cache, compute feature_map(K),
            update the KV state with KV_state += feature_map(K) V^T, 
            and update the cache with the latest K, V
        """
        
        output_attentions = False

        # Get queries, keys, values (projection, rotary embed, repeat_kv)
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        kv_seq_len = k.shape[-2]
        
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Convert `past_key_values` to TKHedgehogWindowAttentionState object
        if past_key_value is not None:
            if not isinstance(past_key_value, TKHedgehogWindowAttentionState):
                past_key_value = TKHedgehogWindowAttentionState()  # This then gets passed thru the rest of the network
                
        # if past_key_value is not None:
        #     if not isinstance(past_key_value, TKHedgehogWindowAttentionState):
        #         past_key_value = TKHedgehogWindowAttentionState()  # Initialize it if it's not the right type

        # When we are generating after prefill
        if q.shape[2] == 1 and kv_seq_len > 1:
            x_len = q.shape[2]  # 1
            f_q = self.feature_map_q(q)
            # Update the KV and K states
            _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                     self.feature_map_k)
            k_cache, v_cache, kv_state, k_state = _kv
            # Sliding window + linear attention decode
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1

            # Softmax attention terms
            a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
            a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
            a_sm   = window_factors * torch.exp(a_sm - a_sm_max)
            sum_sm = a_sm.sum(dim=-1, keepdim=True)

            # Combine with linear attention terms
            y_true = (torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                      + linear_factors * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float()))
            sum_ln = linear_factors * torch.einsum(
                'bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]
            y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype) 
        
        # When we are processing the prefill / prompt
        else:  
            # Use TK-implemented linear + terrace window attention
            tk_pad = 0  # TK needs padding to multiple of 64
            x_len  = q.shape[2]
            if x_len % 64 != 0:
                tk_pad = get_pad_to_multiple(x_len, 64)  
                # Faster this way than padding hidden states
                q = F.pad(q, (0, 0, 0, tk_pad), "constant", 0)
                k = F.pad(k, (0, 0, 0, tk_pad), "constant", 0)
                v = F.pad(v, (0, 0, 0, tk_pad), "constant", 0)

            b, h, l, d = q.shape
            device = q.device
            # tk.hedgehog arguments
            y_true   = torch.zeros(b, h, l, d, dtype=torch.bfloat16, device=device)
            kv_state = torch.zeros(b, h, d, d, dtype=torch.float32, device=device)
            k_state  = torch.zeros(b, h, d, dtype=torch.float32, device=device)
            betas    = F.sigmoid(self.window_factors[0, :, 0, 0].to(dtype=torch.float32))
            alphas   = (1 - betas if self.affine_attention_factors else 
                        torch.ones(betas.shape, dtype=torch.float32, device=device))
            
            q_map = self.feature_map_q.layer
            k_map = self.feature_map_k.layer
            # q_map = self.feature_map_q(q).permute(0, 1, 3, 2)
            # k_map = self.feature_map_k(k).permute(0, 1, 3, 2)
            
            # Saves outputs to y_true, k_state, kv_state, where we fuse:
            # 1. f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)
            # 2. y_true = attention(q, k, f_q, f_k, v)  # b, h, l, d
            # 3. kv_state = torch.einsum(‘bhlf,bhld->bhfd’, 
            #                            f_k[:, :, :-self.window_size], 
            #                            v[:, :, :-self.window_size])  # b, h, f, d
            # 4. k_state = f_k[:, :, :-self.window_size].sum(dim=-2)   # b, h, d
            tk_window_hedgehog_attention(q.contiguous(), k.contiguous(), v.contiguous(), 
                                         y_true, k_state, kv_state, 
                                         q_map.contiguous(), k_map.contiguous(), 
                                         alphas, betas)
            # Save the KV and K states
            past_key_value.update_with_kv(kv_state, k_state.unsqueeze(-2), k, v, x_len, tk_pad, self.layer_idx)
            if tk_pad > 0:
                y_true = y_true[:, :, :x_len]  # b, h, l, d
        
        # Concatenate heads and apply output projection
        y_true = y_true.transpose(1, 2).contiguous().view(b, x_len, self.hidden_size)
        y_true = self.o_proj(y_true)
        attention_weights = None
        return y_true, attention_weights, past_key_value


class TKHedgehogWindowAttentionState(Cache):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36, but `self._seen_tokens` for later versions (4.43)
        self._seen_tokens_by_layer: List[int] = []  # same here, `self.seent_tokens_by_layer` for v4.36
        self.window_size = window_size
        
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

    def update_with_kv(self, kv_state: torch.Tensor, k_state: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, tk_pad: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new KV and K states
        """
        if layer_idx == 0:
            self._seen_tokens += seq_len  # k.shape[2]
        self._seen_tokens_by_layer.append(seq_len)  # (k.shape[2])

        # Initialize KV and K states
        if len(self.decode_k_states) <= layer_idx:  
            self.decode_kv_states.append(kv_state)
            self.decode_k_states.append(k_state)
        else:  # Update KV and K states
            self.decode_kv_states[layer_idx] = self.decode_kv_states[layer_idx] + kv_state
            self.decode_k_states[layer_idx] = self.decode_k_states[layer_idx] + k_state

        self.k_cache.append(k[:, :, -self.window_size:-tk_pad, :])  # keep tokens up to padding
        self.v_cache.append(v[:, :, -self.window_size:-tk_pad, :])

    def update_for_decoding(self, k: torch.Tensor, v: torch.Tensor, 
                            layer_idx: int, feature_map_k: callable) -> None:
        """
        Update the cache for decoding
        """
        k_cache = self.k_cache[layer_idx]
        v_cache = self.v_cache[layer_idx]

        if k_cache.shape[2] == 64:  # hard-code
            k_state = feature_map_k(k_cache[:, :, :1, :])
            v_state = v_cache[:, :, :1, :]
            kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(k.dtype)

            self.decode_kv_states[layer_idx] += kv_state
            self.decode_k_states[layer_idx] += k_state

            self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], k], dim=-2)
            self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], v], dim=-2)
        else:
            # Fill up cache til 64 tokens big
            self.k_cache[layer_idx] = torch.cat([k_cache, k], dim=2)
            self.v_cache[layer_idx] = torch.cat([v_cache, v], dim=2)

        if layer_idx == 0:
            self._seen_tokens += k.shape[-2]
        self._seen_tokens_by_layer[layer_idx] += k.shape[-2]

        return (self.k_cache[layer_idx], self.v_cache[layer_idx],
                self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx])