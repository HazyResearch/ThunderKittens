"""
Install FlashAttention 3:
    cd somewhere-outside-repo
    git clone https://github.com/Dao-AILab/flash-attention/
    cd flash-attention/hopper
    git submodule update --init --recursive
    python -m pip install --upgrade pip wheel setuptools ninja packaging # upgrading these helps faster build
    MAX_JOBS=64 python setup.py install
    or
    MAX_JOBS=64 python setup.py install > build.log 2>&1 &
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
torch.set_printoptions(sci_mode=False)

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark_l2_clear,
    benchmark_no_l2_clear,
    profile,
    clean_print
)

import flash_attn_interface
from _C import TKParallelTensor, tk_mha_fwd_d128  # type: ignore


def tk_ring_attn_fwd(
    Q: torch.Tensor,
    K0: TKParallelTensor,
    K1: TKParallelTensor,
    V0: TKParallelTensor,
    V1: TKParallelTensor,
    L: torch.Tensor,
    L_block: torch.Tensor,
    O: torch.Tensor,
    O_block: torch.Tensor,
    barrier: TKParallelTensor,
    num_comm_sms: int
) -> None:
    for i in range(barrier.local_world_size_):
        tk_mha_fwd_d128(Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier, i, num_comm_sms)


def flash_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor
) -> None:
    Q_flash = Q.transpose(1, 2).contiguous() # We do not care about the speed here
    K_flash = K.transpose(1, 2).contiguous()
    V_flash = V.transpose(1, 2).contiguous()
    O_flash = torch.empty_like(O).transpose(1, 2).contiguous()

    # Returns O (B, N, H, D), LSE (B, H, N), *rest
    flash_attn_interface._flash_attn_forward(
        Q_flash, K_flash, V_flash,
        None, None,  # k_new, v_new
        None,  # qv
        O_flash,  # out
        None, None, None,  # cu_seqlens_q/k/k_new
        None, None,  # seqused_q/k
        None, None,  # max_seqlen_q/k
        None, None, None,  # page_table, kv_batch_idx, leftpad_k,
        None, None, None,  # rotary_cos/sin, seqlens_rotary
        None, None, None,  # q_descale, k_descale, v_descale
        softmax_scale=Q_flash.shape[-1] ** (-0.5),
        causal=False,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    )

    O.copy_(O_flash.transpose(1, 2)) # We do not care about the speed here


def run(
    B: int,
    H: int,
    N: int,
    D: int,
    num_comm_sms: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    torch.manual_seed(42 + local_rank)

    local_K0 = TKParallelTensor(
        (B, H, N // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    local_K1 = TKParallelTensor(
        (B, H, N // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    local_V0 = TKParallelTensor(
        (B, H, N // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    local_V1 = TKParallelTensor(
        (B, H, N // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    local_L = torch.empty(B, H, N // local_world_size, device=f"cuda:{local_rank}", dtype=torch.float32)
    local_L_block = torch.empty_like(local_L)
    local_O = torch.empty(B, H, N // local_world_size, D, device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    local_O_block = torch.empty_like(local_O)

    if check_correctness:
        if local_rank == 0:
            Q = torch.randn((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
            K = torch.randn((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
            V = torch.randn((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
        else:
            Q = torch.empty((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
            K = torch.empty((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
            V = torch.empty((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
        O = torch.empty((B, H, N, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)
        torch.distributed.broadcast(Q, src=0)
        torch.distributed.broadcast(K, src=0)
        torch.distributed.broadcast(V, src=0)
        local_Q = Q.chunk(local_world_size, dim=2)[local_rank].contiguous()
        local_K0.data_.copy_(K.chunk(local_world_size, dim=2)[local_rank].contiguous())
        local_V0.data_.copy_(V.chunk(local_world_size, dim=2)[local_rank].contiguous())
        flash_attn_run = lambda: flash_attn_fwd(Q, K, V, O)

    else:
        local_Q = torch.randn((B, H, N // local_world_size, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)    
        local_K = torch.randn((B, H, N // local_world_size, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)    
        local_V = torch.randn((B, H, N // local_world_size, D), device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=False)    
        local_K0.data_.copy_(local_K)
        local_V0.data_.copy_(local_V)
        del local_K, local_V

    barrier = TKParallelTensor(
        (2, 1024, 1024), # 2 MB is the minimum requirement for multicast object anyways, so no loss here!
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier.data_.zero_()

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    tk_ring_attn_run = lambda: tk_ring_attn_fwd(
        local_Q, local_K0, local_K1, local_V0, local_V1, 
        local_L, local_L_block, local_O, local_O_block,
        barrier, num_comm_sms
    )

    if check_correctness:
        flash_attn_run()
        tk_ring_attn_run()
        check_diff("Ring Attention Diff Comparison", local_O, O.chunk(local_world_size, dim=2)[local_rank])
        torch.distributed.barrier()

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (tk_ring_attn_run())
        profile(combined, num_iters=1, suffix=f"B{B}H{H}N{N}D{D}")

    total_flops = 4 * B * H * N * N * D // local_world_size
    total_tflops = total_flops * 1e-12
    tk_ring_avg_ms = benchmark_no_l2_clear(tk_ring_attn_run, num_warmup_iters, num_iters)
    tk_ring_tflops = total_tflops / (tk_ring_avg_ms * 1e-3)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<TK Ring Attention | world_size={local_world_size} | {B}x{H}x{N}x{D} | num_comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"TK Ring: {tk_ring_avg_ms:.3f} ms | {tk_ring_tflops:.2f} TFLOp/s")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()

    B = 16
    H = 16
    D = 128

    for N in [local_world_size * 768 * (2 ** i) for i in range(1, 7)]:
        for num_comm_sms in [2, 4, 8, 16, 32, 64]: # must be even
            run(B, H, N, D, num_comm_sms, local_rank, local_world_size, check_correctness=False, do_profile=False)

    destroy_distributed_environment()
