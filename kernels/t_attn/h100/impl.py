import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Literal

# Set manual seed for reproducibility
torch.manual_seed(42)

def init_linear_with_constant(layer, const_val=0.1):
    nn.init.constant_(layer.weight, const_val)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize linear layers with constant weights
        init_linear_with_constant(self.to_qkv)
        init_linear_with_constant(self.to_out)

    def forward(self, x, mask = None, attn_bias = None):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        print("Inside Attention forward:")
        print("q shape: ", q.shape)
        print("k shape: ", k.shape)
        print("v shape: ", v.shape)
        print("attn_bias shape: ", attn_bias.shape)
        print("\n")
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if attn_bias is not None:
            dots = dots + attn_bias

        if mask is not None:
            mask = mask[:, None, None, :].expand(-1, h, -1, -1)
            dots = dots.masked_fill(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TriangleAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        node_type: Literal['starting', 'ending'],
        dropout = 0.
    ):
        super().__init__()
        self.need_transpose = node_type == 'ending'

        self.attn = Attention(dim = dim, heads = heads)

        self.dropout = nn.Dropout(dropout)

        self.to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias=False),
            nn.Unflatten(-1, (heads, 1))
        )

        # Initialize linear layer in to_attn_bias with constant weights
        init_linear_with_constant(self.to_attn_bias[0])

    def forward(
        self,
        pairwise_repr: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')
            
        attn_bias = self.to_attn_bias(pairwise_repr)
        attn_bias = rearrange(attn_bias, 'b i j h 1 -> b h i j')
        
        batch, n, _, _ = pairwise_repr.shape
        pairwise_repr = rearrange(pairwise_repr, 'b i j d -> (b i) j d')
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = n)
        
        # none by default
        if mask is not None:
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = n)
            
        print("Inside TriangleAttention forward:")
        print("pairwise repr shape: ", pairwise_repr.shape)
        print("attn_bias shape: ", attn_bias.shape)
        # pairwise representation = (B * N) x N x D
        # attn_bias               = (B * N) x H x N x N
        print("\n")
        
        out = self.attn(
            pairwise_repr,
            mask = None,
            attn_bias = attn_bias
        )

        out = rearrange(out, '(b i) j d -> b i j d', b = batch)

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)

# Set up parameters
batch_size = 2
seq_length = 256
dim = 128
heads = 4

# Create random input tensor
torch.manual_seed(42)
pairwise_repr = torch.randn(batch_size, seq_length, seq_length, dim)

# Initialize TriangleAttention (starting version, not ending)
triangle_attn = TriangleAttention(dim=dim, heads=heads, node_type='starting')

# Perform forward pass
output = triangle_attn(pairwise_repr, mask=None)

# Print output shape and some values for verification
print(f"Input shape: {pairwise_repr.shape}")
print(f"Output shape: {output.shape}")
# print(f"Input first few values: {pairwise_repr[0, 0, 0, :5]}")
# print(f"Output first few values: {output[0, 0, 0, :5]}")