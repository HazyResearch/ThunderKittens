import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark,
    profile,
    clean_print
)

from _C import TKParallelTensor, tk_all_reduce


def run(
    N: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    tensor_tk = TKParallelTensor(
        (N, N), 
        dtype=torch.bfloat16, 
        local_rank=local_rank, 
        local_world_size=local_world_size, 
        multicast=True
    )
    barrier_tk = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    torch.randn((N, N), out=tensor_tk.data_)
    barrier_tk.data_.zero_()
    tensor_nccl = tensor_tk.data_.clone()

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    nccl_func = lambda: torch.distributed.all_reduce(tensor_nccl, op=torch.distributed.ReduceOp.SUM)
    tk_func = lambda: tk_all_reduce(tensor_tk, barrier_tk)

    if check_correctness:
        nccl_func()
        tk_func()
        check_diff("AllReduceSum Diff Comparison", tensor_tk.data_, tensor_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_func(), tk_func())
        profile(combined, num_iters)

    nccl_avg_ms = benchmark(nccl_func, num_warmup_iters, num_iters)
    tk_avg_ms = benchmark(tk_func, num_warmup_iters, num_iters)

    # Although NVLS is used, assume ring all-reduce
    bytes_per_tensor = tensor_tk.data_.numel() * tensor_tk.data_.element_size()
    bytes_per_channel = bytes_per_tensor * 2.0 * (local_world_size - 1) / local_world_size

    nccl_gbps = ((bytes_per_channel * 1e-9) / (nccl_avg_ms * 1e-3))
    tk_gbps = ((bytes_per_channel * 1e-9) / (tk_avg_ms * 1e-3))

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<BF16 AllReduceSum | world_size={local_world_size} | {N}x{N}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_gbps:.2f} GB/s")
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_gbps:.2f} GB/s")

if __name__ == "__main__":
    local_rank, local_world_size, rank, world_size = init_distributed_environment()
    for N in [4096, 8192, 16384, 32768, 65536, 131072]:
        run(N, local_rank, local_world_size, check_correctness=False, do_profile=False)
    destroy_distributed_environment()
