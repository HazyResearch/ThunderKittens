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

from _C import TKParallelTensor, tk_all_to_all


# Flags
CHECK_CORRECTNESS = True
BENCHMARK = True
PROFILE = True
SCATTER_AXIS = 2
GATHER_AXIS = 1


# Taken & modified from long-context-attention (https://github.com/feifeibear/long-context-attention)
def nccl_all_to_all_func(
    output: torch.Tensor,
    input: torch.Tensor,
    local_world_size: int,
    scatter_idx: int,
    gather_idx: int,
) -> torch.Tensor:
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


def tk_all_to_all_func(
    output: TKParallelTensor,
    input: TKParallelTensor,
    barrier: TKParallelTensor,
    scatter_idx: int,
    gather_idx: int,
) -> None:
    tk_all_to_all(output, input, barrier, scatter_idx, gather_idx)


def run(
    N: int,
    H: int,
    D: int,
    scatter_axis: int,
    gather_axis: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    input_tensor_tk = TKParallelTensor(
        (1, N // local_world_size, H, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    torch.randn((1, N // local_world_size, H, D), out=input_tensor_tk.data_)
    output_tensor_tk = TKParallelTensor(
        (1, N, H // local_world_size, D),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
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

    input_tensor_nccl = input_tensor_tk.data_.clone()
    output_tensor_nccl = output_tensor_tk.data_.clone()

    nccl_run = lambda: nccl_all_to_all_func(output_tensor_nccl, input_tensor_nccl, local_world_size, scatter_axis, gather_axis)
    tk_run = lambda: tk_all_to_all_func(output_tensor_tk, input_tensor_tk, barrier_tk, scatter_axis, gather_axis)

    if check_correctness:
        nccl_run()
        tk_run()
        check_diff("AllToAll Diff Comparison", output_tensor_tk.data_, output_tensor_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_run(), tk_run())
        profile(combined, num_iters)

    nccl_avg_ms = benchmark(nccl_run, num_warmup_iters, num_iters)
    tk_avg_ms = benchmark(tk_run, num_warmup_iters, num_iters)

    chunk_size = (N // local_world_size) * (H // local_world_size) * D * 2
    per_rank_comm_size = chunk_size * (local_world_size - 1)
    total_comm_size = per_rank_comm_size * local_world_size

    nccl_gbps = ((per_rank_comm_size * 1e-9) / (nccl_avg_ms * 1e-3))
    tk_gbps = ((per_rank_comm_size * 1e-9) / (tk_avg_ms * 1e-3))

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<BF16 AllToAll | world_size={local_world_size} | {N}x{H}x{D}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_gbps:.2f} GB/s")
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_gbps:.2f} GB/s")

if __name__ == "__main__":
    local_rank, local_world_size, rank, world_size = init_distributed_environment()

    H = 128
    D = 128
    scatter_axis = 2
    gather_axis = 1

    for N in [16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        run(N, H, D, scatter_axis, gather_axis, local_rank, local_world_size, check_correctness=False, do_profile=False)
    destroy_distributed_environment()
