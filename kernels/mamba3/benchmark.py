import os
import sys
import types
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

# Bypass mamba_ssm's __init__.py (which hard-imports compiled CUDA extensions)
# by pre-populating sys.modules with stub packages that have correct __path__
# entries. This lets Python resolve submodule files without executing __init__.py.
_mamba_repo = None
for _candidate in ["/root/mamba-repo", os.path.expanduser("~/mamba-repo")]:
    if os.path.isdir(os.path.join(_candidate, "mamba_ssm")):
        _mamba_repo = _candidate
        break

ref_mamba3_siso_fwd = None
ref_mamba3_siso_combined = None
ref_mamba3_mimo = None

if _mamba_repo is not None:
    if _mamba_repo not in sys.path:
        sys.path.insert(0, _mamba_repo)
    for _pkg, _subdir in [
        ("mamba_ssm", "mamba_ssm"),
        ("mamba_ssm.ops", "mamba_ssm/ops"),
        ("mamba_ssm.ops.triton", "mamba_ssm/ops/triton"),
        ("mamba_ssm.ops.triton.mamba3", "mamba_ssm/ops/triton/mamba3"),
        ("mamba_ssm.ops.tilelang", "mamba_ssm/ops/tilelang"),
        ("mamba_ssm.ops.tilelang.mamba3", "mamba_ssm/ops/tilelang/mamba3"),
    ]:
        if _pkg not in sys.modules:
            _mod = types.ModuleType(_pkg)
            _mod.__path__ = [os.path.join(_mamba_repo, _subdir)]
            _mod.__package__ = _pkg
            sys.modules[_pkg] = _mod

    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd as ref_mamba3_siso_fwd
        print("Successfully imported mamba3_siso_fwd")
    except Exception as e:
        print(f"Failed to import mamba3_siso_fwd: {type(e).__name__}: {e}")

    if ref_mamba3_siso_fwd is None:
        try:
            from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined as ref_mamba3_siso_combined
            print("Successfully imported mamba3_siso_combined")
        except Exception as e:
            print(f"Failed to import mamba3_siso_combined: {type(e).__name__}: {e}")

    try:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as ref_mamba3_mimo
        print("Successfully imported mamba3_mimo (TileLang)")
    except Exception as e:
        print(f"Failed to import mamba3_mimo (TileLang): {type(e).__name__}: {e}")
else:
    print("mamba repo not found; skipping Triton baseline")


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

    def forward(self, x, dt, A, B, C, chunk_size, D=None, use_mimo=False):
        if use_mimo:
            raise RuntimeError("Local Mamba-3 baseline MIMO path is not implemented yet")
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

    def forward(self, x, dt, A, B, C, chunk_size, D=None, use_mimo=False):
        q, k, v, adt, dt_ref, trap, q_bias, k_bias, angles, d_ref = build_reference_mamba3_inputs(
            x, dt, A, B, C, D
        )
        if use_mimo:
            raise RuntimeError("MIMO Triton reference not wired yet")

        # Triton kernel expects ADT/DT/Trap as (batch, nheads, seqlen);
        # build_reference_mamba3_inputs produces them as (batch, seqlen, nheads).
        # Q/K/V are cast to bf16 to match the kernel's internal bf16 storage path.
        q_bf = q.to(torch.bfloat16).contiguous()
        k_bf = k.to(torch.bfloat16).contiguous()
        v_bf = v.to(torch.bfloat16).contiguous()
        q_bias_bf = q_bias.to(torch.bfloat16).contiguous()
        k_bias_bf = k_bias.to(torch.bfloat16).contiguous()
        adt_t = adt.transpose(1, 2).contiguous()
        dt_t = dt_ref.transpose(1, 2).contiguous()
        trap_t = trap.transpose(1, 2).contiguous()

        if ref_mamba3_siso_fwd is not None:
            out = ref_mamba3_siso_fwd(
                q_bf, k_bf, v_bf, adt_t, dt_t, trap_t, q_bias_bf, k_bias_bf, angles,
                D=d_ref, Z=None, chunk_size=chunk_size,
            )
        elif ref_mamba3_siso_combined is not None:
            out = ref_mamba3_siso_combined(
                q_bf, k_bf, v_bf, adt_t, dt_t, trap_t, q_bias_bf, k_bias_bf, angles,
                D=d_ref, Z=None, chunk_size=chunk_size,
            )
        else:
            raise RuntimeError("No mamba3 Triton reference available")

        return out[0] if isinstance(out, tuple) else out


MIMO_RANK = 4
TILELANG_CHUNK_SIZE = 16
TILELANG_ROTARY_DIV = 2


class Mamba3TileLang(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B_gate, C_gate, chunk_size, D=None, use_mimo=False):
        if not use_mimo:
            raise RuntimeError("TileLang reference only wired for MIMO")
        if ref_mamba3_mimo is None:
            raise RuntimeError("mamba3_mimo (TileLang) is not available")

        batch, seqlen, nheads, headdim_v = x.shape
        _, _, ngroups, headdim_qk = C_gate.shape

        q_bf = (
            rearrange(C_gate.expand(-1, -1, MIMO_RANK * ngroups, -1),
                      "b l (r g) d -> b l r g d", r=MIMO_RANK)
            .to(torch.bfloat16)
            .contiguous()
        )
        k_bf = (
            rearrange(B_gate.expand(-1, -1, MIMO_RANK * ngroups, -1),
                      "b l (r g) d -> b l r g d", r=MIMO_RANK)
            .to(torch.bfloat16)
            .contiguous()
        )
        v_bf = x.to(torch.bfloat16).contiguous()

        adt = rearrange(A * dt, "b l h -> b h l").to(torch.float32).contiguous()
        dt_t = rearrange(dt, "b l h -> b h l").to(torch.float32).contiguous()
        trap = torch.zeros_like(adt).to(torch.bfloat16).contiguous()

        q_bias = torch.zeros((nheads, MIMO_RANK, headdim_qk), device=x.device, dtype=torch.float32)
        k_bias = torch.zeros((nheads, MIMO_RANK, headdim_qk), device=x.device, dtype=torch.float32)
        mimo_v = torch.ones((nheads, MIMO_RANK, headdim_v), device=x.device, dtype=torch.float32)
        mimo_out = torch.ones((nheads, MIMO_RANK, headdim_v), device=x.device, dtype=torch.float32)
        mimo_z = torch.zeros((nheads, MIMO_RANK, headdim_v), device=x.device, dtype=torch.float32)

        head_angles = headdim_qk // TILELANG_ROTARY_DIV
        angles = (
            torch.cumsum(dt, dim=1)
            .unsqueeze(-1)
            .expand(batch, seqlen, nheads, head_angles)
            .to(torch.float32)
            .contiguous()
        )

        d_ref = torch.zeros(nheads, device=x.device, dtype=torch.float32) if D is None else D.to(torch.float32).contiguous()
        z = torch.zeros_like(v_bf)

        out = ref_mamba3_mimo(
            q_bf, k_bf, v_bf, adt, dt_t, trap,
            q_bias, k_bias, mimo_v, mimo_z, mimo_out,
            angles, d_ref, z,
            TILELANG_CHUNK_SIZE, TILELANG_ROTARY_DIV,
            torch.bfloat16,
            return_state=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


def build_tk_mamba3_inputs(x, dt, A, B, C):
    q = rearrange(C, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    v = rearrange(x * dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(A * dt, "b l h -> b h l").to(torch.float32).contiguous()

    b = torch.zeros_like(a)
    angle = rearrange(torch.cumsum(dt, dim=1), "b l h -> b h l").to(torch.float32).contiguous()
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
    use_mimo=False,
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
                    y = tk_mamba3(q, k, v, a, b_tk, angle, False)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_tk_mimo":
                    if tk_mamba3 is None:
                        raise RuntimeError("thunderkittens extension is not available")
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = tk_mamba3(q, k, v, a, b_tk, angle, True)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_baseline_siso":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = baseline_method(x, dt, A, B, C, chunk_size, D=None, use_mimo=False)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_baseline_mimo":
                    raise RuntimeError("mamba3_baseline_mimo is not implemented yet")
                elif method_str == "mamba3_triton_siso":
                    if ref_mamba3_siso_fwd is None:
                        raise RuntimeError("mamba3_siso_fwd is not available")
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = triton_method(x, dt, A, B, C, chunk_size, D=None, use_mimo=False)
                    end_events[i].record()
                    torch.cuda.synchronize()
                elif method_str == "mamba3_triton_mimo":
                    raise RuntimeError("mamba3_triton_mimo is not wired yet")
                elif method_str == "mamba3_tilelang_mimo":
                    if ref_mamba3_mimo is None:
                        raise RuntimeError("mamba3_mimo (TileLang) is not available")
                    tilelang_method = Mamba3TileLang()
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = tilelang_method(x, dt, A, B, C, chunk_size, D=None, use_mimo=True)
                    end_events[i].record()
                    torch.cuda.synchronize()
                else:
                    raise AssertionError(f"Unknown method: {method_str}")
            except Exception as e:
                if verbose:
                    import traceback
                    print(f"Error ({type(e).__name__}): {e}")
                    print(traceback.format_exc(), flush=True)
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
        use_mimo=False,
    )
    IMPLEMENTATIONS["mamba3_tk_mimo"] = partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_tk_mimo",
        use_mimo=True,
    )

if ref_mamba3_siso_fwd is not None or ref_mamba3_siso_combined is not None:
    IMPLEMENTATIONS["mamba3_triton_siso"] = partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_triton_siso",
    )

if ref_mamba3_mimo is not None:
    IMPLEMENTATIONS["mamba3_tilelang_mimo"] = partial(
        mamba3_test,
        causal=True,
        is_forwards=True,
        method_str="mamba3_tilelang_mimo",
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
