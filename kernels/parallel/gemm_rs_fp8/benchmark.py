import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

gpu = os.environ.get("GPU", "")
assert gpu == "B200", "GPU must be set to B200"

import torch

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark_l2_clear,
    benchmark_no_l2_clear,
    profile,
    clean_print
)

from _C import TKParallelTensor, matmul_reduce_scatter  # type: ignore


def nccl_matmul_reduce_scatter_func(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> None:
    intermediate = torch.matmul(A.to(torch.bfloat16), B.T.to(torch.bfloat16))
    torch.distributed.reduce_scatter_tensor(C, intermediate, op=torch.distributed.ReduceOp.SUM)


def tk_matmul_reduce_scatter_func(
    A: torch.Tensor,
    B: torch.Tensor,
    C: TKParallelTensor,
    barrier: TKParallelTensor,
) -> None:
    matmul_reduce_scatter(A, B, C, barrier)


def run(
    M: int,
    K: int,
    N: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    A = (torch.randn(M, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25).to(torch.float8_e4m3fn)
    C_tk = TKParallelTensor(
        (M // local_world_size, N),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    C_tk.data_.zero_()
    barrier = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier.data_.zero_()

    C_nccl = torch.zeros(M // local_world_size, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    tk_run = lambda: tk_matmul_reduce_scatter_func(A, B, C_tk, barrier)
    nccl_run = lambda: nccl_matmul_reduce_scatter_func(A, B, C_nccl)

    if check_correctness:
        nccl_run()
        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("Matmul Reduce-Scatter Diff Comparison", C_tk.data_, C_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_run(), tk_run())
        profile(combined, num_iters)

    tk_avg_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)
    total_flops = 2.0 * M * N * K
    total_tflops = total_flops * 1e-12
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<FP8 TP Matmul | world_size={local_world_size} | {M}x{K}x{N}>", print_once=True)
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOp/s")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()

    for N in [2048, 4096, 8192, 16384, 32768]:
        run(N, N // local_world_size, N, local_rank, local_world_size, check_correctness=False, do_profile=False)

    destroy_distributed_environment()
