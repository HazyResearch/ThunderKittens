import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Union, Optional
import copy
from einops import rearrange

import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import triton kernel
from fwd_tri import parallel_based_fwd_kernel_hedgehog

# import fast transformers kernel
try:
    sys.path.append("/var/cr05_data/sim_data/code/release/based/train/")
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported the causal dot product kernel! ")
except:
    print(f"Could not import the causal dot product kernel... ")
    causal_dot_product = None


class TiedHeadMLP(nn.Module):
    """
    Use same linear weights applied to all attention heads
    """
    def __init__(self, 
                 num_heads: int,
                 head_dim: int,     # input dim
                 feature_dim: int,  # output dim
                 dtype: torch.dtype,
                 device: torch.device,
                 skip_connection: bool = True,
                 bias: bool = False,
                 zero_init: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device
        self.skip_connection = skip_connection
        self.zero_init = zero_init
        self.init_weights_()
        
        if self.zero_init: 
            self.zero_init_with_skip_() if self.skip_connection else self.zero_init_()

        if self.skip_connection:
            assertion_fail = f'If self.skip_connection we need self.head_dim == self.feature_dim but self.head_dim is {self.head_dim} != self.feature_dim is {self.feature_dim}'
            assert self.head_dim == self.feature_dim, assertion_fail

    def zero_init_with_skip_(self):
        with torch.no_grad():
            nn.init.zeros_(self.layer.weight)

    def zero_init_(self):
        with torch.no_grad():
            nn.init.eye_(self.layer.weight)

    def init_weights_(self):
        self.layer = nn.Linear(self.head_dim, self.feature_dim, bias=False,
                               dtype=self.dtype, device=self.device)

    def forward(self, x: torch.Tensor):
        """Assume x.shape is b h l d"""
        return x + self.layer(x) if self.skip_connection else self.layer(x)

class UntiedHeadMLP(TiedHeadMLP):
    """
    Use different weights per head
    """
    def init_weights_(self):
        self.layer = nn.Conv1d(in_channels=self.head_dim * self.num_heads,
                               out_channels=self.feature_dim * self.num_heads,
                               kernel_size=1, groups=self.num_heads,
                               bias=False, dtype=self.dtype, device=self.device)

    def zero_init_(self):
        with torch.no_grad():
            nn.init.eye_(self.layer.weight[..., 0])

    def _forward(self, x: torch.Tensor):
        b, h, l, d = x.shape
        x = rearrange(x, 'b h l d -> b (h d) l', h=self.num_heads)
        x = self.layer(x)
        x = rearrange(x, 'b (h d) l -> b h l d', h=self.num_heads)
        return x

    def forward(self, x: torch.Tensor):
        """Assume x.shape is b h l d"""
        return x + self._forward(x) if self.skip_connection else self._forward(x)

class UntiedHeadEinsumMLP(UntiedHeadMLP):
    """
    Alternate implementation with untied heads that uses einsum
    """
    def __init__(self, 
                 normal_init: bool = False, 
                 *args: any, **kwargs: any):
        if normal_init:
            self.nn_init_ = self.normal_init_
        else:
            self.nn_init_ = nn.init.kaiming_uniform_
        super().__init__(*args, **kwargs)
    
    def init_weights_(self):
        self.layer = nn.Parameter(torch.zeros(
            (self.num_heads, self.head_dim, self.feature_dim),
            dtype=self.dtype, device=self.device,
        ))
        self.nn_init_(self.layer)

    def normal_init_(self, layer: torch.Tensor):
        with torch.no_grad():
            for i in range(layer.shape[0]):
                nn.init.normal_(layer[i])

    def zero_init_with_skip_(self):
        with torch.no_grad():
            nn.init.zeros_(self.layer)

    def zero_init_(self):
        with torch.no_grad():
            for i in range(self.layer.shape[0]):
                nn.init.eye_(self.layer[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assume x.shape is b h l d"""
        if self.skip_connection:
            return x + torch.einsum('hdf,bhld->bhlf', self.layer, x)
        return torch.einsum('hdf,bhld->bhlf', self.layer, x)


class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, 
                 input_dim: int,                 
                 temp: int = None,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 mlp: nn.Module = None,
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.temp = 1. if temp is None else temp
        self.eps = eps
        self.mlp = mlp if mlp is not None else nn.Identity()
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return self.mlp(x)

class FullSpaceMap(nn.Module):
    """
    Project positive features to upper and lower "halfspaces"
    """
    def __init__(self, 
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12,
                 **kwargs: any):
        super().__init__()
        self.head_dim_idx = head_dim_idx
        self.eps = eps
        
    def forward(self, x: torch.Tensor, fmap = None):
        return torch.cat([x, -x], dim=self.head_dim_idx).clamp(min=self.eps)

class ExpDim(FeatureMap):
    """
    Feature maps based on applying exp() element- or dimension-wise
    """
    def __init__(self, 
                 fullspace: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.fs_map = FullSpaceMap(**kwargs)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return torch.exp(self.fs_map(x * self.temp))

class SoftmaxDim(ExpDim):
    """
    Compute softmax across fullspace
    """
    def __init__(self, *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
        self.fs_map = None

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x * self.temp
        o = torch.cat([
            torch.softmax( x, dim=self.head_dim_idx),
            torch.softmax(-x, dim=self.head_dim_idx)
        ], dim=self.head_dim_idx).clamp(min=self.eps)
        return o


class Hedgehog(nn.Module):
    def __init__(self, 
                 num_heads: int, 
                 head_dim: int, 
                 feature_dim: int, 
                 input_dim: int, 
                 skip_connection: bool = False, 
                 zero_init: bool = False, 
                 bias: bool = False, 
                 use_triton: bool = False,
                 use_tk: bool = False,
                 use_fast_transformers: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        
        self.skip_connection = skip_connection
        self.zero_init = zero_init
        self.bias = bias
        self.dtype = dtype
        
        self.eps = torch.tensor(1e-12, dtype=self.dtype, device='cuda')
        
        self.use_triton = use_triton
        self.use_tk = use_tk
        self.use_fast_transformers = use_fast_transformers
        assert not (self.use_fast_transformers and self.use_triton), "Cannot use both triton and fast transformers"
        if self.use_fast_transformers: assert causal_dot_product is not None, "Fast transformers kernel not imported"

        layer_kwargs = {
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dtype': self.dtype,
            'device': 'cuda',
        }
        kernel_kwargs = {
            'feature_dim': self.feature_dim,
            'skip_connection': self.skip_connection,
            'zero_init': self.zero_init,
            'bias': self.bias,
        }
        feature_map_kwargs = {
            'input_dim': self.input_dim,
            'eps': self.eps,
            'fullspace': True,
        }

        self.learned_kernel = UntiedHeadEinsumMLP(**layer_kwargs, **kernel_kwargs)
        self.feature_map_q = SoftmaxDim(mlp=self.learned_kernel, **feature_map_kwargs)
        self.feature_map_k = copy.deepcopy(self.feature_map_q)
        
        self.BS_k_d = min(128, triton.next_power_of_2(self.head_dim))
    

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_scale: bool, use_norm: bool): 
        q, k = self.feature_map_q(q), self.feature_map_k(k)
            
        o = torch.empty_like(v)
        z = torch.empty(q.size(0), q.size(1), q.size(2), dtype=q.dtype, device=q.device)
        
        BS_q_n = 128
        BS_kv_n = 32
        
        BS_k_d = min(128, triton.next_power_of_2(k.shape[-1]))
        BS_v_dv = min(128, triton.next_power_of_2(v.shape[-1]))
        BS_k_d, BS_v_dv = max(BS_k_d, 16), max(BS_v_dv, 16)
        
        D = q.shape[-1] # head_dim
        DV = v.shape[-1]  # feature_dim
        
        NK = triton.cdiv(D, BS_k_d)
        NV = triton.cdiv(DV, BS_v_dv)
        
        num_stages = 2
        num_warps = 4
        
        grid = (NK * NV, triton.cdiv(q.size(2), BS_q_n), q.size(0) * q.size(1))
        
        scale = 1.0
        if use_scale:
            scale = D ** -0.5
               
        if self.use_triton:        
            # below is the pytorch equiv of the triton kernel
            # in terms of style of computation 

            # # compute linear attention
            # cumsum_matrix = torch.tril(torch.ones((q.size(2), q.size(2)), device=q.device, dtype=q.dtype))
            # A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) * cumsum_matrix
            # o = torch.einsum("bhnm,bhme->bhne", A_qk, v)
            
            print("Using qkv triton style computation")
            
            print("q.shape: ", q.shape)
            print("k.shape: ", k.shape)
            print("v.shape: ", v.shape)
            print("\n")
            
            parallel_based_fwd_kernel_hedgehog[grid](
                q, k, v, o,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                q.size(0), q.size(1), q.size(2),
                scale,
                BS_q_n=BS_q_n, BS_kv_n=BS_kv_n, 
                BS_k_d=BS_k_d, 
                BS_v_dv=BS_v_dv, 
                DK=D, 
                DV=DV,
                num_warps=num_warps,
                num_stages=num_stages
            )
                
            if use_norm:
                q, k = q.unsqueeze(-2), k.unsqueeze(-2)
                z = (q * k.cumsum(dim=2)).sum(dim=-1) + self.eps
                return o / z

        elif self.use_fast_transformers:
            print("Using qkv fast transformers style computation")
            
            print("q.shape: ", q.shape)
            print("k.shape: ", k.shape)
            print("v.shape: ", v.shape)
            print("\n")
            
            v = causal_dot_product(
                q.contiguous().to(dtype=torch.float32), 
                k.contiguous().to(dtype=torch.float32),
                v.contiguous().to(dtype=torch.float32),
            )
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl", 
                    q.to(dtype=torch.float32), 
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps
            )
            y = v * z[..., None]
            return y

        else:
            print("Using qkv linear attention style computation")
            
            print("q.shape: ", q.shape)
            print("k.shape: ", k.shape)
            print("v.shape: ", v.shape)
            print("\n")
            
            if use_scale:
                q = q * (q.shape[-1] ** -0.5)

            # compute linear attention
            q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
            
            o = (q * (k * v).cumsum(dim=2)).sum(dim=-1)
            
            # apply normalization
            if use_norm:
                z = (q * k.cumsum(dim=2)).sum(dim=-1) + self.eps
                return o / z
        
        return o