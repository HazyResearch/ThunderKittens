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

from _C import TKParallelTensor, tk_all_gather


def nccl_all_gather_func(
    output: torch.Tensor,
    input: torch.Tensor,
    local_world_size: int
) -> None:
    """
    NCCL AllGather on tensor dimension.
    """
    input_reshaped = input.movedim(-1, 0).contiguous()
    output_reshaped = torch.empty(
        (input_reshaped.shape[0] * local_world_size, *input_reshaped.shape[1:]),
        device=input.device, dtype=input.dtype
    )
    torch.distributed.all_gather_into_tensor(output_reshaped, input_reshaped)
    output.copy_(output_reshaped.movedim(0, -1))


def tk_all_gather_func(
    output: TKParallelTensor,
    input: TKParallelTensor,
    barrier: TKParallelTensor
) -> None:
    tk_all_gather(output, input, barrier)


def run(
    N: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    input_tensor_tk = TKParallelTensor(
        (N, N // local_world_size), 
        dtype=torch.bfloat16, 
        local_rank=local_rank, 
        local_world_size=local_world_size, 
        multicast=False
    )
    torch.randn((N, N // local_world_size), out=input_tensor_tk.data_)
    output_tensor_tk = TKParallelTensor(
        (N, N), 
        dtype=torch.bfloat16, 
        local_rank=local_rank, 
        local_world_size=local_world_size, 
        multicast=True
    )
    output_tensor_tk.data_.zero_()
    barrier_tk = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier_tk.data_.zero_()

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    input_tensor_nccl = input_tensor_tk.data_.clone()
    output_tensor_nccl = output_tensor_tk.data_.clone()

    nccl_run = lambda: nccl_all_gather_func(output_tensor_nccl, input_tensor_nccl, local_world_size)
    tk_run = lambda: tk_all_gather_func(output_tensor_tk, input_tensor_tk, barrier_tk)

    if check_correctness:
        nccl_run()
        tk_run()
        check_diff("AllGather Diff Comparison", output_tensor_tk.data_, output_tensor_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_run(), tk_run())
        profile(combined, num_iters)

    nccl_avg_ms = benchmark(nccl_run, num_warmup_iters, num_iters)
    tk_avg_ms = benchmark(tk_run, num_warmup_iters, num_iters)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<BF16 AllGather | world_size={local_world_size} | {N}x{N}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms")
    clean_print(f"TK: {tk_avg_ms:.3f} ms")

if __name__ == "__main__":
    local_rank, local_world_size, rank, world_size = init_distributed_environment()
    for N in [4096, 8192, 16384, 32768, 65536, 131072]:
        run(N, local_rank, local_world_size, check_correctness=False, do_profile=False)
    destroy_distributed_environment()
