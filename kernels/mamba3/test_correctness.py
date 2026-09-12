import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MAMBA_ROOT = REPO_ROOT / "tk_offical" / "mamba"
if OFFICIAL_MAMBA_ROOT.exists():
    sys.path.insert(0, str(OFFICIAL_MAMBA_ROOT))

try:
    from _C import mamba3
    print("Successfully imported thunderkittens")
except ImportError:
    print('Failed to import thunderkittens; Please "pip setup.py install".')
    mamba3 = None

try:
    from mamba_ssm.ops.triton.mamba3 import mamba3_siso_fwd as ref_mamba3_siso_fwd
    print("Successfully imported mamba3_siso_fwd from cloned official mamba repo")
except ImportError:
    print("Failed to import mamba3_siso_fwd from cloned official mamba repo")
    ref_mamba3_siso_fwd = None

from baselines.ssd_minimal import (
    apply_trapezoidal_to_decay,
    expand_groups_to_heads,
    pairwise_trapezoidal_scale,
    segsum,
    trapezoidal_scale,
)


warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd(args...)` is deprecated..*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd(args...)` is deprecated*")


def ssd_trap_only_discrete(X, A, B, C, Trap, block_len, initial_states=None):
    """
    Trap-only discrete SSD reference matching the current TK kernel shape contract.

    Arguments:
        X:    (batch, length, n_heads, d_head)
        A:    (batch, length, n_heads)
        B:    (batch, length, n_heads, d_state)
        C:    (batch, length, n_heads, d_state)
        Trap: (batch, length, n_heads)
    Return:
        Y:          (batch, length, n_heads, d_head)
        final_state (batch, n_chunks, d_head, d_state)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype == Trap.dtype
    assert X.shape[1] % block_len == 0

    X, A, B, C, Trap = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
        for x in (X, A, B, C, Trap)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    Trap = rearrange(Trap, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    local_decay = torch.exp(segsum(A))
    trap_scale = trapezoidal_scale(A, Trap)
    trap_pairwise = pairwise_trapezoidal_scale(trap_scale)
    local_decay = apply_trapezoidal_to_decay(local_decay, trap_pairwise)

    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, local_decay, X)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)

    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    y = rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
    return y, final_state


def build_tk_inputs(x, dt, A, B, C, trap_beta):
    q = rearrange(C, "b l g d -> b g l d").to(torch.bfloat16).contiguous()
    k = rearrange(B, "b l g d -> b g l d").to(torch.bfloat16).contiguous()
    v = rearrange(x * dt.unsqueeze(-1), "b l h d -> b h l d").to(torch.bfloat16).contiguous()
    a = rearrange(A * dt, "b l h -> b h l").to(torch.float32).contiguous()
    angle = torch.zeros_like(a)
    return q, k, v, a, trap_beta.contiguous(), angle


def build_official_siso_inputs(x, dt, A, B, C, D=None):
    batch, seqlen, nheads, _ = x.shape
    q = expand_groups_to_heads(C, nheads).to(torch.float32).contiguous()
    k = expand_groups_to_heads(B, nheads).to(torch.float32).contiguous()
    v = x.to(torch.float32).contiguous()
    adt = rearrange(A * dt, "b l h -> b h l").to(torch.float32).contiguous()
    dt_ref = rearrange(dt, "b l h -> b h l").to(torch.float32).contiguous()
    trap = torch.zeros_like(adt)
    q_bias = torch.zeros((nheads, q.shape[-1]), device=x.device, dtype=torch.float32)
    k_bias = torch.zeros((nheads, k.shape[-1]), device=x.device, dtype=torch.float32)
    angles = torch.zeros((batch, seqlen, nheads, q.shape[-1] // 2), device=x.device, dtype=torch.float32)
    d_ref = None if D is None else D.to(torch.float32).contiguous()
    return q, k, v, adt, dt_ref, trap, q_bias, k_bias, angles, d_ref


def print_errors(name, x_true, x):
    diff = (x_true.to(torch.float32) - x.to(torch.float32)).abs()
    avg_diff = diff.mean()
    max_diff = diff.max()
    avg_magnitude = x_true.to(torch.float32).abs().mean().clamp_min(1e-6)
    avg_error = avg_diff / avg_magnitude
    print(
        f"{name:18s} avg_diff={avg_diff:.6f}, max_diff={max_diff:.6f}, "
        f"avg_mag={avg_magnitude:.6f}, avg_error={avg_error * 100:.2f}%"
    )
    return avg_diff, max_diff, avg_error


def test_correctness(num_iters=10, batch=2, nheads=16, seqlen=1024, chunk_size=64, headdim=64):
    if mamba3 is None:
        raise RuntimeError("TK mamba3 extension is not available")

    assert seqlen % chunk_size == 0, "seqlen must be divisible by chunk_size"

    dtype = torch.float32
    device = "cuda"
    ngroups = 1
    dstate = 64

    official_smoke_done = False
    for i in range(num_iters):
        x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
        A = -torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))
        B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
        C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
        D = torch.randn(nheads, dtype=dtype, device=device)
        trap_beta = (0.05 * torch.randn(batch, nheads, seqlen, dtype=torch.float32, device=device)).contiguous()

        q, k, v, a, b, angle = build_tk_inputs(x, dt, A, B, C, trap_beta)
        torch.cuda.synchronize()
        y_tk = mamba3(q, k, v, a, b, angle, False)
        torch.cuda.synchronize()
        y_tk = rearrange(y_tk, "b h l d -> b l h d").contiguous().to(torch.float32)

        has_nan = torch.isnan(y_tk).any().item()
        has_inf = torch.isinf(y_tk).any().item()
        if has_nan or has_inf:
            raise AssertionError(f"Iteration {i}: TK output has NaN={has_nan} Inf={has_inf}")

        X_ref = x * dt.unsqueeze(-1)
        A_ref = (A * dt).to(torch.float32).contiguous()
        Trap_ref = rearrange(trap_beta, "b h l -> b l h").to(torch.float32).contiguous()
        B_ref = expand_groups_to_heads(B, nheads).to(torch.float32).contiguous()
        C_ref = expand_groups_to_heads(C, nheads).to(torch.float32).contiguous()

        y_ref, _ = ssd_trap_only_discrete(X_ref, A_ref, B_ref, C_ref, Trap_ref, chunk_size)

        _, _, avg_error = print_errors(f"iter={i}", y_ref, y_tk)
        if avg_error > 0.10:
            raise AssertionError(f"Iteration {i}: average relative error too high ({avg_error * 100:.2f}%)")

        if ref_mamba3_siso_fwd is not None and not official_smoke_done:
            q_ref, k_ref, v_ref, adt_ref, dt_ref, trap_ref, q_bias, k_bias, angles_ref, d_ref = build_official_siso_inputs(
                x, dt, A, B, C, D
            )
            y_official = ref_mamba3_siso_fwd(
                q_ref,
                k_ref,
                v_ref,
                adt_ref,
                dt_ref,
                trap_ref,
                q_bias,
                k_bias,
                angles_ref,
                D=d_ref,
                Z=None,
                chunk_size=chunk_size,
            )
            if not torch.isfinite(y_official).all():
                raise AssertionError("Official mamba3_siso_fwd smoke test produced non-finite output")
            print(f"official_siso_smoke output shape={tuple(y_official.shape)} finite=True")
            official_smoke_done = True

        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_correctness()
