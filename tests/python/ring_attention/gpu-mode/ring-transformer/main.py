import sys
from typing import Callable, Optional
import os
import torch
from torch.optim import AdamW
from torch import nn
from einops import rearrange

import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from flash_attn import flash_attn_qkvpacked_func


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, causal: bool, attn_fn: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.heads = heads
        self.causal = causal
        self.attn_fn = attn_fn

    def forward(self, x):
        # (batch_size, seqlen, hidden_dim)
        x = self.norm(x)

        x = self.qkv_proj(x)
        # (batch_size, seqlen, (3 * nheads * headdim))

        x = rearrange(x, "B S (n nheads headdim) -> B S n nheads headdim", n=3, nheads=self.heads)
        # (batch_size, seqlen, 3, nheads, headdim)

        #x = flash_attn_qkvpacked_func(
        x = self.attn_fn(x, causal=self.causal)

        # (batch_size, seqlen, nheads, headdim)
        x = rearrange(x, "B S nheads headdim -> B S (nheads headdim)")

        # (batch_size, seqlen, hidden_dim)
        x = self.out_proj(x)

        return x


class Layer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, causal: bool, attn_fn: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, heads, causal, attn_fn)
        self.ffn = FeedForward(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, out_dim: int, num_layers: int, hidden_dim: int, heads: int, mlp_dim: int, causal: bool, attn_fn: Callable):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(Layer(hidden_dim=hidden_dim, heads=heads, mlp_dim=mlp_dim, causal=causal, attn_fn=attn_fn))
        self.output_head = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x):
        for l in self.layers:
           x = l(x)
        x = self.output_head(x)
        return x


def broadcast_model(model, source=0):
    for param in model.parameters():
        dist.broadcast(param.data, src=source)


def cross_check():
    B,S,D = 2,4,16

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    dist.broadcast(x, src=0)

    t1 = Transformer(num_layers=2, out_dim=D, hidden_dim=D, heads=2, mlp_dim=D*4, attn_fn=flash_attn_qkvpacked_func)
    t1.to(device=device, dtype=dtype)

    broadcast_model(t1, source=0)

    t2 = Transformer(num_layers=2, out_dim=D, hidden_dim=D, heads=2, mlp_dim=D*4, attn_fn=ring_flash_attn_qkvpacked_func)
    t2.to(device=device, dtype=dtype)

    for a,b in zip(t1.parameters(), t2.parameters()):
        b.data.copy_(a)

    y1 = t1(x)

    local_x = x.chunk(world_size, dim=1)[rank].detach().clone()

    y2 = t2(local_x)
    print(f"rank={rank}, shape={y2.shape}")
    #print(f"rank {rank}: {y}")

    if rank == 0:
        print("diff: ", y1.chunk(world_size, dim=1)[rank] - y2)


def test_single_device_training_without_ring_attn(device: Optional[torch.DeviceObjType]):
    """Test constant batch training with single GPU without ring-attention, using original `flash_attn_qkvpacked_func`"""
    B,S,D = 2,100,128

    dtype = torch.bfloat16
    device = device or torch.device(f"cuda:0")

    print(f"device={device}, dtype={dtype}")

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    y = torch.randn_like(x)

    t = Transformer(num_layers=2, out_dim=D, hidden_dim=D, heads=2, mlp_dim=D*4, causal=True, attn_fn=flash_attn_qkvpacked_func)
    t.to(device=device, dtype=dtype)

    optmizer = AdamW(params = t.parameters(), lr=0.001)
    loss = nn.MSELoss()

    for i in range(500):

        out = t(x)

        l = loss(out, y)

        if i % 10 == 0:
            print(f'iter={i}, loss={l}')

        t.zero_grad()
        l.backward()
        optmizer.step()


def main():
    if "WORLD_SIZE" not in os.environ:
        sys.exit("env var WORLD_SIZE not set, run with torchrun")

    num_layers = 2
    out_dim = 128
    hidden_dim = 128
    mlp_dim = hidden_dim * 4
    heads = 4

    B,S,D = 2,100,128       # batch_size, seq_len, embedding_dim

    steps = 200

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    print(f"rank={rank}, world_size={world_size}, device={device}, dtype={dtype}")

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    y = torch.randn_like(x)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    t = Transformer(
        num_layers=num_layers,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        mlp_dim=mlp_dim,
        causal=True,
        attn_fn=ring_flash_attn_qkvpacked_func
    )
    t.to(device=device, dtype=dtype)
    broadcast_model(t, source=0)

    optmizer = AdamW(params = t.parameters(), lr=0.01)
    loss = nn.MSELoss()

    local_x = x.chunk(world_size, dim=1)[rank].detach().clone()
    local_y = y.chunk(world_size, dim=1)[rank].detach().clone()

    for i in range(steps):

        out = t(local_x)

        l = loss(out, local_y)

        if i % 10 == 0:
            print(f'iter={i}, rank={rank}, loss={l}')

        t.zero_grad()
        l.backward()

        for p in t.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        optmizer.step()


if __name__ == '__main__':
    main()
