"""
Linear attention in Based. 
"""
import math

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple

from train.src.generation import InferenceParams

import sys
sys.path.append("../../../")
sys.path.append("/home/bfs/simran/clean2/ThunderKittens/examples/based/linear_attn_forward/H100/")

try:
    from train.csrc.causal_dot_prod import causal_dot_product  # linear attention cuda kernel
    print(f"Successfully imported the causal dot product kernel! ")
except:
    print(f"Could not import the causal dot product kernel... ")
    causal_dot_product = None

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    print(f"Successfully imported the FLA triton kernels! ")
except:
    print(f"Could not import the FLA triton kernels... ")

try:
    import lin_attn as mod
    print(f"Successfully imported TK based_H100 kernel")
except:
    mod = None
    print("Could not import TK based_H100 kernel --- Not needed.")

try:
    import based_inference as mod_inf 
    print(f"Successfully imported TK based_inference decode kernel")
except:
    mod_inf = None
    print("Could not import TK based_inference decode kernel")

        
class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x
    
class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            **kwargs: any
        ):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)     # Note that this is the feature dimension.
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        # SE: raising to power 0 is a hacky way to get ones without calling torch.ones
        # which is incompatible with cuda graph caching 
        return torch.cat(
            [x[..., :1] ** 0, x / self.rrd, x2 / self.rd], 
            dim=-1
        )

class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        feature_map: FeatureMap, 
        l_max: int = 2048,
        feature_dim: int = 16,
        head_dim: int = None,
        num_heads: int = 16,
        eps: float = 1e-12,
        layer_idx: int = None,
        parallel_implementation: str="quadratic",
        inference_implementation: str="default",
        recurrent_impl: str="default",
        silent=True,
        inference_bs: int=1,
        use_decay_proj: bool = False,
        **kwargs
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.l_max = l_max
        self.eps = eps
        self.parallel_implementation = parallel_implementation

        # set dimension 
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads if head_dim is None else head_dim      
        self.feature_dim = feature_dim

        # initialize projections and feature map
        self.feature_map = feature_map
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        # optional input-dependent decay from Based
        self.use_decay_proj = use_decay_proj
        if self.use_decay_proj:
            # we have an n x n matrix of decay values - lets make it input-dependent
            self.decay_proj = nn.Linear(self.d_model, self.num_heads, bias=False)

        self.inference_implementation = inference_implementation
        self.recurrent_impl = recurrent_impl
        self.silent=silent
        if self.inference_implementation == "tk":
            self.d_state = 320
            bs = inference_bs
            assert bs is not None, print(f"for TK, need to pass in the batch size")
            self.y = torch.empty((bs, self.num_heads, self.l_max, self.head_dim), dtype=torch.bfloat16, device='cuda')
            self.y_rec = torch.empty((bs * self.num_heads, self.head_dim), dtype=torch.bfloat16, device='cuda')
            
            self.kv_state_a2 = torch.empty((bs, self.num_heads, self.feature_dim*self.feature_dim, self.head_dim), dtype=torch.bfloat16, device='cuda')
            self.kv_state_a1 = torch.empty((bs, self.num_heads, self.head_dim, self.feature_dim), dtype=torch.bfloat16, device='cuda')
            self.kv_state_a0 = torch.empty((bs, self.num_heads, self.head_dim), dtype=torch.bfloat16, device='cuda')
            
            self.k_state_a2 = torch.empty((bs, self.num_heads, self.feature_dim*self.feature_dim), dtype=torch.bfloat16, device='cuda')
            self.k_state_a1 = torch.empty((bs, self.num_heads, self.feature_dim), dtype=torch.bfloat16, device='cuda')
            self.k_state_a0 = torch.ones((bs, self.num_heads), dtype=torch.bfloat16, device='cuda') * self.l_max

            if self.recurrent_impl == "default": 
                self.padding = torch.zeros(bs, self.num_heads, self.d_state-self.expanded_size(), dtype=torch.bfloat16, device='cuda')
                self.qk_padding = torch.zeros(bs*self.num_heads, self.d_state-self.expanded_size(), dtype=torch.bfloat16, device='cuda')
                self.kv_padding = torch.zeros(bs, self.num_heads, self.d_state-self.expanded_size(), self.head_dim, dtype=torch.bfloat16, device='cuda')

        
    def forward(self, 
        hidden_states: torch.Tensor,
        inference_params: InferenceParams = None,
        decay: Optional[Tuple[torch.Tensor]] = None,
        *args: any, 
        **kwargs: any
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """

        decay, decay_recurrent = decay if decay is not None else (None, None)

        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if inference_params is None:
            return self.parallel_forward(hidden_states, q, k, v, decay=decay)
        else:
            impl_choice = self.inference_implementation

            # check if we are doing parallel prefill or recurrent generation
            if inference_params.seqlen_offset > 0: 
                # recurrent
                kv_state, k_state = self._get_inference_cache(inference_params)
                q, k = self.feature_map(q), self.feature_map(k)
                return self.recurrent_forward(hidden_states, kv_state, k_state, q, k, v, decay=decay_recurrent, impl_choice=impl_choice).to(hidden_states.dtype)
            else:  
                # prefill
                y = self.parallel_forward(hidden_states, q, k, v, decay=decay, impl_choice=impl_choice)
                if impl_choice != "default" and impl_choice == "tk" and self.recurrent_impl == "default":
                    if self.layer_idx < 2 and not self.silent: print("recurrent state tk")
                    
                    kv_state = torch.concat([self.kv_state_a0.unsqueeze(-2), self.kv_state_a1.transpose(2,3), self.kv_state_a2, self.kv_padding], dim=-2)
                    k_state = torch.concat([self.k_state_a0.unsqueeze(-1), self.k_state_a1,  self.k_state_a2, self.padding], dim=-1)

                elif impl_choice != "default" and impl_choice == "tk":
                    
                    kv_state = torch.concat([
                        self.kv_state_a0.unsqueeze(-1), 
                        self.kv_state_a1, 
                        self.kv_state_a2.transpose(2,3)
                    ], dim=-1)[:, :, None]
                    k_state = torch.concat([
                        self.k_state_a0.unsqueeze(-1),
                        self.k_state_a1, 
                        self.k_state_a2
                    ], dim=-1)[:, :, None, None]
                else:
                    k = self.feature_map(k)
                    if decay is not None:
                        if len(decay.shape) == 3:
                            decay = decay.unsqueeze(0)
                        k_decay = decay[:, :, l - 1 , :l, None] * k
                        kv_state = torch.einsum("bhnd,bhnf->bhfd", k_decay, v)[:, :, None]
                    else:
                        kv_state = torch.einsum("bhnd,bhnf->bhfd", k, v)[:, :, None]
                    k_state = k.sum(dim=2)[:, :, None, None]

                if self.layer_idx in inference_params.key_value_memory_dict:
                    # # update the state in-place when graph caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                else: 
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)
                return y

    def parallel_forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor=None, impl_choice: str="default"):
        b, l, _ = x.size()

        if impl_choice != "default" and impl_choice == "tk":
            if self.layer_idx <= 2 and not self.silent: print(f"parallel tk")
            b, h, l, d = v.shape
            b, h, n, D = q.shape
            dt = q.dtype
            q = torch.cat([q, torch.zeros(b, h, self.l_max - l, D, dtype=q.dtype, device=q.device)], dim=2)
            k = torch.cat([k, torch.zeros(b, h, self.l_max - l, D, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(b, h, self.l_max - l, d, dtype=v.dtype, device=v.device)], dim=2)

            add_scale, add_norm, output_state = 1, 1, 1
            mod.based_fwd_tk(
                int(add_scale),int(add_norm),int(output_state),
                q.to(dtype=torch.bfloat16).contiguous(),k.to(dtype=torch.bfloat16).contiguous(),v.to(dtype=torch.bfloat16).contiguous(), self.y,
                self.kv_state_a2,self.kv_state_a1,self.kv_state_a0,
                self.k_state_a2,self.k_state_a1
            )

            y = self.y[:, :, :l]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "quadratic":
            q, k = self.feature_map(q), self.feature_map(k)
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
            if decay is not None and self.use_decay_proj:
                decay = decay[:, :l, :l]
                if len(decay.shape) == 3:
                    decay = decay.unsqueeze(0)
                # (b l d) --> (b, l, h)
                dt_out = self.decay_proj(x) 
                assert decay.shape[2] >= l, f"decay matrix shape too short"
                # (b, h, l, 1) * (1, h, l, l)
                decay_mat = dt_out.transpose(1,2).unsqueeze(-1) * decay   
                A_qk = A_qk * decay_mat
            else:
                A_qk = torch.tril(A_qk)        
            y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
            y = y * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "linear": 
            assert decay is None, "Decay not for this view"
            
            q, k = self.feature_map(q), self.feature_map(k)
            v = causal_dot_product(q.contiguous().to(dtype=torch.float32), k.contiguous().to(dtype=torch.float32),v.contiguous().to(dtype=torch.float32),)
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl", 
                    q.to(dtype=torch.float32), 
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps
            )
            y = v * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "fla_parallel":
            """ 
            Computes both the feature map and causal dot products.
            Booleans are for the denominator and the normalization 
            """
            assert decay is None, "Decay not for this view"
            y = parallel_based(q, k, v, True, True)
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "fla_chunk":
            """ 
            Computes both the feature map and causal dot products.
            Booleans are for the denominator and the normalization 
            """
            assert decay is None, "Decay not for this view"
            y = fused_chunk_based(q, k, v, True, True)
            y = rearrange(y, 'b h l d -> b l (h d)')

        else: 
            raise ValueError(f"Parallel implementation {self.parallel_implementation} not supported")

        return self.out_proj(y.to(x.dtype))

    
    def recurrent_forward(self, hidden_states: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor=None, impl_choice: str="default"):
        """
        Compute linear attention with recurrent view
        -> Assume q.shape is (b, h, 1, d); k and v.shape are (b, h, l, d)
        """
        if self.layer_idx <= 2 and not self.silent: print(f"recurrent")
        b, h, l, d = q.shape
        b, h, l, dv = v.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'

        if impl_choice != "default" and impl_choice == "tk" and self.recurrent_impl == "default":
            if self.layer_idx <= 2 and not self.silent: print(f"recurrent tk")

            q = rearrange(q, 'b h 1 d -> (b h) d')
            k = rearrange(k, 'b h 1 d -> (b h) d')
            v = rearrange(v, 'b h 1 d -> (b h) d')

            q, k = torch.cat([q, self.qk_padding], dim=-1), torch.cat([k, self.qk_padding], dim=-1)
            mod_inf.based_step(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), 
                rearrange(kv_state, 'b h f d -> (b h) f d').to(torch.bfloat16), 
                rearrange(k_state, 'b h d -> (b h) d').to(torch.bfloat16), 
                self.y_rec
            )
            y = self.y_rec.view(b, h, v.shape[-1]).unsqueeze(2).to(q.dtype)

        else:
            # Expand dims for broadcasting to compute linear attention
            q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

            kv_state += k[:, :, -1:] * v[:, :, -1:]
            if decay is not None: 
                kv_state.copy_(torch.einsum("h,bhldf->bhldf", decay, kv_state) + k[:, :, -1:].to(torch.float32) * v[:, :, -1:].to(torch.float32))
            else:
                kv_state += k[:, :, -1:].to(torch.float32) * v[:, :, -1:].to(torch.float32)
            k_state  += k[:, :, -1:].to(torch.float32)

            # Compute linear attention
            num = (q * kv_state).sum(dim=-1)

            if self.use_decay_proj:
                dt_out = self.decay_proj(hidden_states).squeeze(1)
                num = num * dt_out[..., None, None]
                
            y = num.to(torch.float32) / ((q.to(torch.float32) * k_state.to(torch.float32)).sum(dim=-1) + self.eps)

        y = rearrange(y, 'b h l d -> b l (h d)').to(q.dtype)
        return self.out_proj(y)
 
    
    def expanded_size(self):
        return self.feature_dim ** 2 + self.feature_dim + 1
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        if self.inference_implementation == 'tk' and self.recurrent_impl == "default":
            kv_shape = (batch_size, self.num_heads, self.d_state, self.head_dim, )
            k_shape = (batch_size, self.num_heads, self.d_state)
            kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
            k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        else:
            kv_shape = (
                batch_size, self.num_heads, 1, self.head_dim, self.expanded_size()
            )
            k_shape = (
                batch_size, self.num_heads, 1, 1, self.expanded_size()
            )
            kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
            k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        return (kv_state, k_state)
     
    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]

