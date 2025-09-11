import os
from typing import Callable

import torch
torch.set_printoptions(sci_mode=False)


def init_distributed_environment():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == local_world_size, "multi-node runs are not supported"
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=local_rank,
        rank=rank,
        world_size=world_size
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.random.manual_seed(local_rank)

    return local_rank, local_world_size, rank, world_size


def destroy_distributed_environment():
    torch.distributed.destroy_process_group()


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor):
    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<{name}>", print_once=True)
    clean_print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    clean_print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    clean_print(f"Mean:      {A.abs().mean().item():.10f}")
    clean_print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    clean_print(f"Max:       {A.abs().max().item():.10f}")
    clean_print(f"Ref max:   {A_ref.abs().max().item():.10f}")


def benchmark(
    func: Callable,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        func()
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iters

    return avg_ms


def profile(
    func: Callable,
    num_iters: int = 10
) -> None:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True
    ) as profiler:
        for _ in range(num_iters):
            func()

    # Export to Chrome trace format
    trace_filename = f"rank_{int(os.environ.get("LOCAL_RANK", 0))}_trace.json"
    profiler.export_chrome_trace(trace_filename)
    clean_print(f"Profiler trace exported to {trace_filename}")


def clean_print(*args, **kwargs):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if kwargs.pop("print_once", False):
        if local_rank == 0:
            print(*args, **kwargs)
        torch.distributed.barrier()
    else:
        for i in range(local_world_size):
            if i == local_rank:
                print(f"[Rank {i}]", *args, **kwargs)
            torch.distributed.barrier()
