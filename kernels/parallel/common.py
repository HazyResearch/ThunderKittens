import os
from typing import Callable
from time import perf_counter

import torch
torch.set_printoptions(sci_mode=False)

_env = {}


def init_distributed_environment():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == local_world_size, "multi-node runs are not supported"
    assert rank == local_rank, "multi-node runs are not supported"
    torch.distributed.init_process_group(
        backend="nccl",
        device_id=local_rank,
        rank=rank,
        world_size=world_size
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.random.manual_seed(local_rank)

    _env["local_rank"] = local_rank
    _env["local_world_size"] = local_world_size
    _env["rank"] = rank
    _env["world_size"] = world_size

    return local_rank, local_world_size


def destroy_distributed_environment():
    torch.distributed.destroy_process_group()


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor, single: bool = False):
    if single:
        print(f"===============================================================================")
        print(f"<{name}>")
        print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        print(f"Mean:      {A.abs().mean().item():.10f}")
        print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        print(f"Max:       {A.abs().max().item():.10f}")
        print(f"Ref max:   {A_ref.abs().max().item():.10f}")
    else:
        clean_print(f"===============================================================================", print_once=True)
        clean_print(f"<{name}>", print_once=True)
        clean_print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        clean_print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        clean_print(f"Mean:      {A.abs().mean().item():.10f}")
        clean_print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        clean_print(f"Max:       {A.abs().max().item():.10f}")
        clean_print(f"Ref max:   {A_ref.abs().max().item():.10f}")


def benchmark_no_l2_clear(
    func: Callable,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    single: bool = False, # for the sake of consistency with benchmark_l2_clear
    use_events: bool = True # only valid if using default stream
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    if use_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / num_iters

    else:
        start_time = perf_counter()
        for _ in range(num_iters):
            func()
        torch.cuda.synchronize()
        end_time = perf_counter()
        avg_ms = (end_time - start_time) * 1000 / num_iters

    return avg_ms


def benchmark_l2_clear(
    func: Callable,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    single: bool = False,
    use_events: bool = True
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    l2_cache = torch.empty(1024 * 1024 * 128 // 2, dtype=torch.bfloat16, device=f"cuda:{_env['local_rank']}" if not single else "cuda")

    if use_events:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            l2_cache.random_(0, 100)
            start_events[i].record()
            func()
            end_events[i].record()
        torch.cuda.synchronize()

        times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
        total_ms = sum(times)
        avg_ms = total_ms / num_iters

    else:
        times = []
        for i in range(num_iters):
            l2_cache.random_(0, 100)
            torch.cuda.synchronize()
            start_time = perf_counter()
            func()
            torch.cuda.synchronize()
            end_time = perf_counter()
            times.append(end_time - start_time)
        avg_ms = (sum(times) / num_iters) * 1000

    return avg_ms


def profile(
    func: Callable,
    num_iters: int = 5,
    suffix: str = ""
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
    trace_filename = f"rank_{int(os.environ.get('LOCAL_RANK', 0))}{'_' if suffix else ''}{suffix}_trace.json"
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
