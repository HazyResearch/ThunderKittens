"Based on: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py"
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat


@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _update_step(
    # Pointers to matrices
    kv_state_ptr, v_ptr, k_ptr, q_ptr, out_ptr,
    # Matrix dimensions
    dim, dstate,
    # Strides
    stride_kv_state_batch, stride_kv_state_dim, stride_kv_state_dstate,
    stride_v_batch, stride_v_dim,
    stride_k_batch, stride_k_dstate,
    stride_q_batch, stride_q_dstate,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    kv_state_ptr += pid_b * stride_kv_state_batch
    v_ptr += pid_b * stride_v_batch
    k_ptr += pid_b * stride_k_batch
    q_ptr += pid_b * stride_q_batch

    out_ptr += pid_b * stride_out_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    kv_state_ptrs = kv_state_ptr + (offs_m[:, None] * stride_kv_state_dim + offs_n[None, :] * stride_kv_state_dstate)
    v_ptrs = v_ptr + offs_m * stride_v_dim
    k_ptrs = k_ptr + offs_n * stride_k_dstate
    q_ptrs = q_ptr + offs_n * stride_q_dstate
    out_ptrs = out_ptr + offs_m * stride_out_dim

    kv_state = tl.load(kv_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    V = tl.load(v_ptrs, mask=offs_m < dim, other=0.0)
    K = tl.load(k_ptrs, mask=offs_n < dstate, other=0.0)
    Q = tl.load(q_ptrs, mask=offs_n < dstate, other=0.0)

    kv_state = kv_state + K[None, :] * V[:, None]
    tl.store(kv_state_ptrs, kv_state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    num = tl.sum(kv_state * Q[None, :], axis=1)
    tl.store(out_ptrs, num, mask=offs_m < dim)


def lin_attn_step(
    kv_state, 
    v, k, q
):
    """
    Argument:
        kv state: (batch, dim, dstate)
        v: (batch, dim)
        k: (batch, dstate)
        q: (batch, dstate)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = kv_state.shape
    # print(f"{batch=}, {dim=}, {dstate=}")
    # breakpoint()
    assert v.shape == (batch, dim)
    assert k.shape == (batch, dstate)
    assert q.shape == k.shape

    out = torch.empty_like(v)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch)
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <=  16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    BLOCK_SIZE_M, num_warps = (4, 8)

    with torch.cuda.device(v.device.index):
        _update_step[grid](
            kv_state, v, k, q, out,
            dim, dstate,
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
            v.stride(0), v.stride(1),
            k.stride(0), k.stride(1),
            q.stride(0), q.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return out
