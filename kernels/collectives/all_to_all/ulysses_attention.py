"""
A separate file for testing the ulysses-style SP with all2all kernel
    torchrun --nproc_per_node=8 ulysses_attention.py

To install required packages:
    pip install cuda-python==12.8.* nvidia-cutlass-dsl==4.2.* flash_attn==2.8.*
"""

import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)

import cuda.bindings.driver as cuda
import cutlass
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark,
    profile,
    clean_print
)

from _C import TKParallelTensor, tk_all_to_all # type: ignore


# Flags
CHECK_CORRECTNESS = True
BENCHMARK = True
PROFILE = False


# Taken & modified from long-context-attention (https://github.com/feifeibear/long-context-attention)
def nccl_all2all(
    output: torch.Tensor,
    input: torch.Tensor,
    local_world_size: int,
    scatter_idx: int,
    gather_idx: int,
) -> None:
    if scatter_idx == 2 and gather_idx == 1:
        B, N_per_rank, H, D = input.shape
        N = N_per_rank * local_world_size
        H_per_rank = H // local_world_size

        input_t = input.view(B, N_per_rank, local_world_size, H_per_rank, D).permute(2, 1, 0, 3, 4).contiguous()
        
        output_t = torch.empty_like(input_t)
        torch.distributed.all_to_all_single(output_t, input_t)

        output.copy_(output_t.permute(2, 0, 1, 3, 4).reshape(B, N_per_rank * local_world_size, H_per_rank, D))
    
    elif scatter_idx == 1 and gather_idx == 2:
        B, N, H_per_rank, D = input.shape
        H = H_per_rank * local_world_size
        N_per_rank = N // local_world_size

        input_t = input.view(B, local_world_size, N_per_rank, H_per_rank, D).permute(1, 3, 2, 0, 4).contiguous()

        output_t = torch.empty_like(input_t)
        torch.distributed.all_to_all_single(output_t, input_t)

        output.copy_(output_t.permute(3, 2, 0, 1, 4).reshape(B, N_per_rank, H_per_rank * local_world_size, D))
    
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def tk_all2all(
    output: TKParallelTensor,
    input: TKParallelTensor,
    barrier: TKParallelTensor,
    scatter_idx: int,
    gather_idx: int,
) -> None:
    tk_all_to_all(output, input, barrier, scatter_idx, gather_idx)


compile_funcs = {}
def flash_attn_fwd_raw(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float,
    O: torch.Tensor,
    L: torch.Tensor
) -> None:
    B, N, H, D_qk = Q.shape
    _, _, _, D_vo = V.shape

    Q_cute = from_dlpack(Q.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    K_cute = from_dlpack(K.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    V_cute = from_dlpack(V.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    O_cute = from_dlpack(O.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    L_cute = from_dlpack(L.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=L.ndim - 1)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Arbitrary key to use the global dict
    if "FA4" not in compile_funcs:
        _fa_fwd = FlashAttentionForwardSm100(
            D_qk,
            D_vo,
            qhead_per_kvhead=1,
            is_causal=True,
            is_local=False,
            pack_gqa=False,
            is_persistent=False # always false if causal
        )
        compile_funcs["FA4"] = cutlass.cute.compile(
            _fa_fwd, Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
            None, None, None, None, None, None, None, None, None
        )

    compile_funcs["FA4"](
        Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
        None, None, None, None, None, None, None, None, None
    )


def nccl_ulysses_attn(
    Q_sp: torch.Tensor,
    Q_tp: torch.Tensor,
    K_sp: torch.Tensor,
    K_tp: torch.Tensor,
    V_sp: torch.Tensor,
    V_tp: torch.Tensor,
    O_sp: torch.Tensor,
    O_tp: torch.Tensor,
    L: torch.Tensor,
    local_world_size: int
) -> None:
    softmax_scale = 1 / math.sqrt(Q_sp.shape[-1])
    nccl_all2all(Q_tp, Q_sp, local_world_size, 2, 1)
    nccl_all2all(K_tp, K_sp, local_world_size, 2, 1)
    nccl_all2all(V_tp, V_sp, local_world_size, 2, 1)
    flash_attn_fwd_raw(Q_tp, K_tp, V_tp, softmax_scale, O_tp, L)
    nccl_all2all(O_sp, O_tp, local_world_size, 1, 2)


def tk_ulysses_attn(
    Q_sp: TKParallelTensor,
    Q_tp: TKParallelTensor,
    K_sp: TKParallelTensor,
    K_tp: TKParallelTensor,
    V_sp: TKParallelTensor,
    V_tp: TKParallelTensor,
    O_sp: TKParallelTensor,
    O_tp: TKParallelTensor,
    L: torch.Tensor,
    barrier: TKParallelTensor
) -> None:
    softmax_scale = 1 / math.sqrt(Q_sp.data_.shape[-1])
    tk_all2all(Q_tp, Q_sp, barrier, 2, 1)
    tk_all2all(K_tp, K_sp, barrier, 2, 1)
    tk_all2all(V_tp, V_sp, barrier, 2, 1)
    flash_attn_fwd_raw(Q_tp.data_, K_tp.data_, V_tp.data_, softmax_scale, O_tp.data_, L)
    tk_all2all(O_sp, O_tp, barrier, 1, 2)


def run(
    N: int,
    H: int,
    D: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    Q_sp_tk = TKParallelTensor(
        (1, N // local_world_size, H, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    torch.randn((1, N // local_world_size, H, D), out=Q_sp_tk.data_)
    Q_tp_tk = TKParallelTensor(
        (1, N, H // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    Q_tp_tk.data_.zero_()
    K_sp_tk = TKParallelTensor(
        (1, N // local_world_size, H, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    torch.randn((1, N // local_world_size, H, D), out=K_sp_tk.data_)
    K_tp_tk = TKParallelTensor(
        (1, N, H // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    K_tp_tk.data_.zero_()
    V_sp_tk = TKParallelTensor(
        (1, N // local_world_size, H, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    torch.randn((1, N // local_world_size, H, D), out=V_sp_tk.data_)
    V_tp_tk = TKParallelTensor(
        (1, N, H // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    V_tp_tk.data_.zero_()
    O_sp_tk = TKParallelTensor(
        (1, N // local_world_size, H, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    O_sp_tk.data_.zero_()
    O_tp_tk = TKParallelTensor(
        (1, N, H // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    O_tp_tk.data_.zero_()
    L_tk = torch.zeros(1, H, N, dtype=torch.float32, device=f"cuda:{local_rank}")
    barrier_tk = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier_tk.data_.zero_()

    Q_sp_nccl = Q_sp_tk.data_.clone()
    Q_tp_nccl = Q_tp_tk.data_.clone()
    K_sp_nccl = K_sp_tk.data_.clone()
    K_tp_nccl = K_tp_tk.data_.clone()
    V_sp_nccl = V_sp_tk.data_.clone()
    V_tp_nccl = V_tp_tk.data_.clone()
    O_sp_nccl = O_sp_tk.data_.clone()
    O_tp_nccl = O_tp_tk.data_.clone()
    L_nccl = L_tk.clone()

    nccl_run = lambda: nccl_ulysses_attn(
        Q_sp_nccl, Q_tp_nccl, K_sp_nccl, K_tp_nccl, V_sp_nccl, V_tp_nccl, O_sp_nccl, O_tp_nccl, L_nccl, local_world_size
    )
    tk_run = lambda: tk_ulysses_attn(
        Q_sp_tk, Q_tp_tk, K_sp_tk, K_tp_tk, V_sp_tk, V_tp_tk, O_sp_tk, O_tp_tk, L_tk, barrier_tk
    )

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    if check_correctness:
        nccl_run()
        tk_run()
        check_diff("Ulysses Attention Diff Comparison", O_sp_tk.data_, O_sp_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_run(), tk_run())
        profile(combined, num_iters)

    tk_avg_ms = benchmark(tk_run, num_warmup_iters, num_iters)
    nccl_avg_ms = benchmark(nccl_run, num_warmup_iters, num_iters)

    total_flops = 2 * (H // local_world_size) * (N * (N + 1) / 2) * (D + D)
    total_tflops = total_flops * 1e-12

    nccl_tflops = total_tflops / (nccl_avg_ms * 1e-3)
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<Ulysses Attention | world_size={local_world_size} | {N}x{H}x{D}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_tflops:.2f} TFLOP/s")
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    local_rank, local_world_size, rank, world_size = init_distributed_environment()

    H = 128
    D = 128

    for N in [16384, 32768, 65536, 131072, 262144, 524288]:
        run(N, H, D, local_rank, local_world_size, check_correctness=False, do_profile=False)

    destroy_distributed_environment()
    del compile_funcs
