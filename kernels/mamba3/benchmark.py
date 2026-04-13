import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from functools import partial

from kernels.common.benchmark_harness import run_benchmark_main

try:
    from mamba3.baselines.ssd_minimal import (
        apply_rotary_last_dim,
        apply_trapezoidal_to_decay,
        expand_groups_to_heads,
        pairwise_trapezoidal_scale,
        segsum,
        trapezoidal_scale,
    )
except Exception:
    from baselines.ssd_minimal import (
        apply_rotary_last_dim,
        apply_trapezoidal_to_decay,
        expand_groups_to_heads,
        pairwise_trapezoidal_scale,
        segsum,
        trapezoidal_scale,
    )

try:
    from _C import mamba3 as tk_mamba3
    print("Successfully imported thunderkittens")
except ImportError:
    print('Failed to import thunderkittens; Please "pip setup.py install".')
    tk_mamba3 = None

try:
    from mamba_ssm.ops.triton.mamba3 import mamba3_siso_fwd as ref_mamba3_siso_fwd
    print("Successfully imported mamba3_siso_fwd")
except ImportError:
    print("Failed to import mamba3_siso_fwd from mamba repo; Please 'pip install mamba_ssm'.")
    ref_mamba3_siso_fwd = None

try:
    from mamba_ssm.ops.modules import mamba3 as ref_mamba3_module
    print("Successfully imported mamba3.py")
except ImportError:
    print("Failed to import mamba3.py from mamba repo; Please 'pip install mamba_ssm'.")
    ref_mamba3_module = None


################### Mamba 3 ####################


def get_flops(b: int, n: int, dv: int, h: int, causal: bool = False) -> int:
    """Calculate FLOPs for Mamba-3 operation."""
    chunk_length = 64
    state_dimension = 64
    ngroups = 1

    num_chunks = n // chunk_length
    seqlen = n
    nheads = h
    head_dimension = dv

    flops = 0
    flops += seqlen * nheads
    flops += seqlen * nheads * chunk_length
    flops += seqlen * nheads * chunk_length
    flops += 2 * num_chunks * ngroups * (chunk_length * chunk_length * state_dimension)
    flops += num_chunks * nheads * (chunk_length * chunk_length)
    flops += 2 * num_chunks * nheads * (chunk_length * chunk_length * head_dimension)
    flops += num_chunks * nheads * chunk_length * 2
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)
    flops += nheads * num_chunks * num_chunks * 2
    flops += num_chunks * nheads * head_dimension * state_dimension
    flops += nheads * num_chunks * chunk_length
    flops += 2 * (num_chunks * nheads * chunk_length * head_dimension * state_dimension)
    flops += num_chunks * nheads * chunk_length * head_dimension
    flops += num_chunks * chunk_length * nheads * head_dimension
    return b * flops


def get_inputs_mamba(dtype: torch.dtype, b: int, h: int, n: int, dv: int, verbose: bool = True):
    """Generate inputs for Mamba-3 computation."""
    torch.manual_seed(42)

    batch, seqlen = b, n
    chunk_size = 64
    dim = h * dv
    headdim = dv
    ngroups = 1
    dstate = 64
    device = "cuda"

    if verbose:
        print(f"Dimensions: batch={batch}, seqlen={seqlen}, heads={h}, dim={dim}, headdim={headdim}")

    x = torch.randn(batch, seqlen, h, headdim, dtype=torch.float32, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, h, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = -torch.exp(torch.rand(h, dtype=torch.float32, device=device)).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(h, dtype=torch.float32, device=device)

    return x, dt, A, B, C, D, chunk_size, torch.float32


def build_reference_mamba3_inputs(x, dt, A, B, C, D=None):
    batch, seqlen, nheads, _ = x.shape
    q = expand_groups_to_heads(C, nheads).to(torch.float32).contiguous()
    k = expand_groups_to_heads(B, nheads).to(torch.float32).contiguous()
    v = x.to(torch.float32).contiguous()
    adt = (A * dt).to(torch.float32).contiguous()
    dt_ref = dt.to(torch.float32).contiguous()
    trap = torch.zeros_like(adt)
    q_bias = torch.zeros((nheads, q.shape[-1]), device=x.device, dtype=torch.float32)
    k_bias = torch.zeros((nheads, k.shape[-1]), device=x.device, dtype=torch.float32)
    angles = (
        torch.cumsum(dt, dim=1)
        .unsqueeze(-1)
        .expand(batch, seqlen, nheads, q.shape[-1] // 2)
        .to(torch.float32)
        .contiguous()
    )
    d_ref = None if D is None else D.to(torch.float32).contiguous()
    return q, k, v, adt, dt_ref, trap, q_bias, k_bias, angles, d_ref


class Mamba3Baseline(torch.nn.Module):
    """Local baseline using the helper functions in baselines/ssd_minimal.py."""

    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, chunk_size, D=None):
        q, k, v, adt, _dt_ref, trap, _q_bias, _k_bias, angles, _d_ref = build_reference_mamba3_inputs(
            x, dt, A, B, C, D
        )

        trap_scale_tensor = trapezoidal_scale(adt.transpose(1, 2), trap.transpose(1, 2)).transpose(1, 2)
        q = apply_rotary_last_dim(q, angles)
        k = apply_rotary_last_dim(k, angles)

        decay = torch.exp(segsum(adt.transpose(1, 2))).transpose(-1, -2)
        trap_pairwise = pairwise_trapezoidal_scale(trap_scale_tensor.transpose(1, 2)).transpose(-1, -2)
        decay = apply_trapezoidal_to_decay(decay, trap_pairwise)

        scores = torch.einsum("blhd,bshd->bhls", q, k)
        scores = scores * decay
        y = torch.einsum("bhls,bshd->blhd", scores, v)
        return y


class Mamba3Triton(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, chunk_size, D=None):
        q, k, v, adt, dt_ref, trap, q_bias, k_bias, angles, d_ref = build_reference_mamba3_inputs(
            x, dt, A, B, C, D
        )

        if ref_mamba3_siso_fwd is None:
            raise RuntimeError("mamba3_siso_fwd is not available")
        return ref_mamba3_siso_fwd(
            q,
            k,
            v,
            adt,
            dt_ref,
            trap,
            q_bias,
            k_bias,
            angles,
            D=d_ref,
            Z=None,
            chunk_size=chunk_size,
        )


def build_tk_mamba3_inputs(x, dt, A, B, C):
    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(x * dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(A * dt, "b l h -> b h l").to(torch.float32).contiguous()

    # Keep the current TK interface stable while the kernel is trap-only. The angle tensor is
    # still passed through for API compatibility, but the current TK kernel ignores it.
    b = torch.zeros_like(a)
    angle = torch.zeros_like(a)
    return q, k, v, a, b, angle


def mamba3_test(
    dtype,
    b,
    h,
    n,
    dv,
    causal,
    is_forwards,
    method_str,
    num_iters=10,
    verbose=True,
    torch_compile=False,
    **kwargs,
):
    baseline_method = Mamba3Baseline()
    triton_method = Mamba3Triton()
    if torch_compile and method_str.startswith("mamba3_triton"):
        try:
            triton_method = torch.compile(triton_method)
        except Exception as e:
            if verbose:
                print(f"Could not compile triton_method: {e}")

    for _stage in ["warmup", "timed"]:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            x, dt, A, B, C, D, chunk_size, _ = get_inputs_mamba(dtype, b, h, n, dv, verbose)
            q, k, v, a, b_tk, angle = build_tk_mamba3_inputs(x, dt, A, B, C)
            a = a.requires_grad_()

            try:
                if method_str == "mamba3_tk_siso":
                    if tk_mamba3 is None:
                        raise RuntimeError("thunderkittens extension is not available")
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = tk_mamba3(q, k, v, a, b_tk, angle)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_baseline_siso":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = baseline_method(x, dt, A, B, C, chunk_size, D=None)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_triton_siso":
                    if ref_mamba3_siso_fwd is None:
                        raise RuntimeError("mamba3_siso_fwd is not available")
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = triton_method(x, dt, A, B, C, chunk_size, D=None)
                    end_events[i].record()
                    torch.cuda.synchronize()
                else:
                    raise AssertionError(f"Unknown method: {method_str}")
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None, -1

            torch.cuda.empty_cache()

    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    tot = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iters
    return y, tot


IMPLEMENTATIONS = {
    "mamba3_baseline_siso": partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_baseline_siso",
    ),
}

if tk_mamba3 is not None:
    IMPLEMENTATIONS["mamba3_tk_siso"] = partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_tk_siso",
    )

if ref_mamba3_siso_fwd is not None:
    IMPLEMENTATIONS["mamba3_triton_siso"] = partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_triton_siso",
    )

NAME = "MAMBA 3"

b = 16
h = 16
dv = 64


if __name__ == "__main__":
    run_benchmark_main(
        name=NAME,
        implementations=IMPLEMENTATIONS,
        get_flops=get_flops,
        b=b,
        h=h,
        dv=dv,
        verbose=True,
        torch_compile=False,
    )
