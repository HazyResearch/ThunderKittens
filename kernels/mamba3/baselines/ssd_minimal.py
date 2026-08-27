# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal helper functions for Mamba-3-style SSD baselines.

This file intentionally provides accurate and most simplest version of Mamba 3's SSD
"""

import torch
from einops import repeat


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable lower-triangular segment sum.

    Args:
        x: Tensor with sequence dimension in the last axis.

    Returns:
        Tensor with shape ``(..., T, T)`` containing causal segment sums and
        ``-inf`` in the strictly upper triangle.
    """
    t = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=t)
    lower_exclusive = torch.tril(
        torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1
    )
    x = x.masked_fill(~lower_exclusive, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    lower_inclusive = torch.tril(
        torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0
    )
    return x_segsum.masked_fill(~lower_inclusive, -torch.inf)


def trapezoidal_scale(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the simple trapezoidal correction factor used by the current TK kernel.

    Args:
        a: Discretized decay term with shape ``(..., L)``.
        b: Trapezoidal correction term with shape ``(..., L)``.

    Returns:
        Tensor with shape ``(..., L)`` where entry ``[..., i]`` is
        ``1 + b[..., i + 1] * exp(-a[..., i + 1])`` for ``i < L - 1`` and ``1``
        for the final position.
    """
    if a.shape != b.shape:
        raise ValueError(f"Expected matching shapes for a and b, got {a.shape} and {b.shape}")
    scale = torch.ones_like(a)
    scale[..., :-1] = 1.0 + b[..., 1:] * torch.exp(-a[..., 1:])
    return scale


def pairwise_trapezoidal_scale(scale: torch.Tensor) -> torch.Tensor:
    """Lift per-position trapezoidal factors into a causal pairwise matrix.

    Args:
        scale: Tensor with shape ``(..., L)``.

    Returns:
        Tensor with shape ``(..., L, L)`` whose lower triangle contains the
        column-wise trapezoidal correction factors and zeros above the diagonal.
    """
    t = scale.size(-1)
    pair = repeat(scale, "... l -> ... r l", r=t)
    row_idx = torch.arange(t, device=scale.device)[:, None]
    col_idx = torch.arange(t, device=scale.device)[None, :]
    causal = row_idx >= col_idx
    strict_lower = row_idx > col_idx
    pair = torch.where(strict_lower, pair, torch.ones_like(pair))
    return pair.masked_fill(~causal, 0)


def apply_trapezoidal_to_decay(decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply trapezoidal factors to an already-exponentiated causal decay matrix."""
    if decay.shape != scale.shape:
        raise ValueError(
            f"Expected matching shapes for decay and scale, got {decay.shape} and {scale.shape}"
        )
    return decay * scale


def rotary_sin_cos(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sine and cosine factors for rotary-style updates."""
    return torch.sin(angle), torch.cos(angle)


def apply_rotary_pair(x0: torch.Tensor, x1: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a single pair of coordinates using a shared angle tensor."""
    sin, cos = rotary_sin_cos(angle)
    y0 = cos * x0 - sin * x1
    y1 = sin * x0 + cos * x1
    return y0, y1


def apply_rotary_last_dim(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply RoPE-style pair rotation across the last dimension of ``x``.

    Args:
        x: Tensor with an even last dimension.
        angle: Tensor broadcastable to ``x[..., ::2]``.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    if x.size(-1) % 2 != 0:
        raise ValueError(f"Expected even last dimension for rotary pairs, got {x.size(-1)}")
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even, y_odd = apply_rotary_pair(x_even, x_odd, angle)
    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y


def expand_groups_to_heads(x: torch.Tensor, nheads: int) -> torch.Tensor:
    """Expand a grouped tensor ``(batch, seqlen, ngroups, dstate)`` to heads."""
    ngroups = x.shape[2]
    if nheads % ngroups != 0:
        raise ValueError(f"nheads={nheads} must be divisible by ngroups={ngroups}")
    heads_per_group = nheads // ngroups
    return x.repeat_interleave(heads_per_group, dim=2)
