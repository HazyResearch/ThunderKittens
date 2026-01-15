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

from _C import TKParallelTensor, all_gather_matmul  # type: ignore


def nccl_all_gather_matmul_func(
    A_local: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> None:
    torch.distributed.all_gather_into_tensor(A, A_local)
    torch.matmul(A.to(torch.bfloat16), B.T.to(torch.bfloat16), out=C)


def tk_all_gather_matmul_func(
    A: TKParallelTensor,
    B: torch.Tensor,
    C: torch.Tensor,
    barrier: TKParallelTensor,
    num_comm_sms: int
) -> None:
    all_gather_matmul(A, B, C, barrier, num_comm_sms)


def run(
    M: int,
    K: int,
    N: int,
    num_comm_sms: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    A_local = (torch.randn(M // local_world_size, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25).to(torch.float8_e4m3fn)
    A_tk = TKParallelTensor(
        (M, K),
        dtype=torch.float8_e4m3fn,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    # Note: this is ONLY for convenience; separating A and A_local does NOT affect performance
    A_tk.data_[local_rank * (M // local_world_size):(local_rank + 1) * (M // local_world_size)] = A_local
    A_nccl = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device=f"cuda:{local_rank}")
    B = (torch.randn(N, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25).to(torch.float8_e4m3fn)
    C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    C_nccl = torch.zeros(M, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
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

    tk_run = lambda: tk_all_gather_matmul_func(A_tk, B, C_tk, barrier, num_comm_sms)
    nccl_run = lambda: nccl_all_gather_matmul_func(A_local, A_nccl, B, C_nccl)

    if check_correctness:
        nccl_run()
        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("All-Gather Matmul Diff Comparison", C_tk, C_nccl)

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
    clean_print(f"<BF16 TP Matmul | world_size={local_world_size} | {M}x{K}x{N} | num_comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOp/s")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()

    for N in [2048, 4096, 8192, 16384, 32768]:
        for num_comm_sms in [2, 4, 8, 16, 32, 64]:
            run(N, N, N // local_world_size, num_comm_sms, local_rank, local_world_size, check_correctness=False, do_profile=False)

    destroy_distributed_environment()
