import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

from _C import TKParallelTensor, moe_dispatch_gemm  # type: ignore


def tk_moe_dispatch_gemm_func(
    *, # to prevent naive mistakes
    inputs_local: TKParallelTensor, # (B * S // local_world_size, H)
    inputs_gathered: torch.Tensor, # (num_padded_local_tokens, H)
    weights: torch.Tensor, # (num_experts_per_dev, H, I)
    outputs: torch.Tensor, # (num_padded_local_tokens, I)
    padded_tokens_per_expert: torch.Tensor, # (num_experts,) of int32
    pull_dispatch_indices: torch.Tensor, # (num_padded_local_tokens, 2)
    barrier: TKParallelTensor,
    num_comm_sms: int,
    num_padded_local_tokens: int
) -> None:
    moe_dispatch_gemm(
        inputs_local,
        inputs_gathered,
        weights,
        outputs,
        padded_tokens_per_expert,
        pull_dispatch_indices,
        barrier,
        num_comm_sms,
        num_padded_local_tokens
    )


@torch.no_grad()
def torch_moe_dispatch_gemm_func(
    *, # to prevent naive mistakes
    inputs_local: TKParallelTensor, # (B * S // local_world_size, H)
    inputs_gathered: torch.Tensor, # (num_padded_local_tokens, H)
    weights: torch.Tensor, # (num_experts_per_dev, H, I)
    outputs: torch.Tensor, # (num_padded_local_tokens, I)
    padded_tokens_per_expert: torch.Tensor, # (num_experts,) of int32
    pull_dispatch_indices: torch.Tensor, # (num_padded_local_tokens, 2)
    local_rank: int,
    local_world_size: int
) -> None:
    inputs_full = torch.empty(local_world_size, inputs_local.shape[0], inputs_local.shape[1], device=inputs_local.device, dtype=inputs_local.dtype)
    torch.distributed.all_gather_into_tensor(inputs_full, inputs_local)
    for local_token_idx in range(pull_dispatch_indices.shape[0]):
        src_dev_idx = pull_dispatch_indices[local_token_idx, 0]
        src_token_idx = pull_dispatch_indices[local_token_idx, 1]
        if src_dev_idx >= 0 and src_token_idx >= 0:
            inputs_gathered[local_token_idx] = inputs_full[src_dev_idx, src_token_idx]

    num_experts = padded_tokens_per_expert.shape[0]
    num_experts_per_dev = num_experts // local_world_size
    expert_offset = num_experts_per_dev * local_rank

    tokens_start = 0
    for local_expert_idx in range(num_experts_per_dev):
        tokens_end = tokens_start + padded_tokens_per_expert[local_expert_idx + expert_offset]
        torch.matmul(
            inputs_gathered[tokens_start:tokens_end],
            weights[local_expert_idx],
            out=outputs[tokens_start:tokens_end]
        )
        tokens_start = tokens_end


def run(
    B: int, # batch size
    S: int, # sequence length
    H: int, # hidden size
    I: int, # expert hidden size
    num_experts: int,
    top_k: int,
    num_comm_sms: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 1,
    num_iters: int = 5,
    check_correctness: bool = False,
    do_profile: bool = False
) -> None:
    device = f"cuda:{local_rank}"
    num_init_tokens_per_dev = B * S // local_world_size
    num_experts_per_dev = num_experts // local_world_size

    # Generate input tensors
    torch.random.manual_seed(42 + local_rank)
    inputs_local = torch.randn(num_init_tokens_per_dev, H, device=device, dtype=torch.bfloat16) / H ** 0.5
    inputs_local_tk = TKParallelTensor(
        (num_init_tokens_per_dev, H),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    inputs_local_tk.data_.copy_(inputs_local)
    weights = torch.randn(num_experts_per_dev, H, I, device=device, dtype=torch.bfloat16) / H ** 0.5

    # Generate MoE routing tensors
    if local_rank == 0:
        # Use the same routing mechanism as Comet (cf. gen_moe_gating_args in flux/testing/moe_utils.py)
        routing_weights = torch.rand(num_experts, device=device, dtype=torch.float32)
        chosen_experts = torch.multinomial(routing_weights.repeat(B * S, 1), top_k, replacement=False).to(torch.int32)
        tokens_per_expert = torch.bincount(chosen_experts.view(-1), minlength=num_experts).to(torch.int32)
    else:
        chosen_experts = torch.empty(B * S, top_k, device=device, dtype=torch.int32)
        tokens_per_expert = torch.empty(num_experts, device=device, dtype=torch.int32)
    torch.distributed.broadcast(chosen_experts, 0)
    torch.distributed.broadcast(tokens_per_expert, 0)

    # Calculate tokens count
    padded_tokens_per_expert = (tokens_per_expert + 127) // 128 * 128
    num_local_tokens = tokens_per_expert[local_rank * num_experts_per_dev:(local_rank + 1) * num_experts_per_dev].sum()
    num_padded_local_tokens = padded_tokens_per_expert[local_rank * num_experts_per_dev:(local_rank + 1) * num_experts_per_dev].sum()
    num_max_tokens = tokens_per_expert.reshape(local_world_size, num_experts_per_dev).sum(dim=1).amax()
    num_padded_max_tokens = padded_tokens_per_expert.reshape(local_world_size, num_experts_per_dev).sum(dim=1).amax()

    # Generate dispatch schedule (*baseline benchmark pre-generates the schedule as well)
    clean_print("Generating dispatch schedule...", print_once=True)
    expert_start = num_experts_per_dev * local_rank
    expert_end = num_experts_per_dev * (local_rank + 1)
    token_index_per_expert = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device), 
        torch.cumsum(padded_tokens_per_expert[expert_start:expert_end - 1], dim=0, dtype=torch.int32)
    ], dim=0)
    pull_dispatch_indices = torch.full((num_padded_local_tokens, 2), -1, dtype=torch.int32, device=device) # 2 for (src_dev_idx, src_token_idx)
    chosen_experts_cpu = chosen_experts.to("cpu")
    for i in range(local_world_size):
        src_dev_idx = (i + local_rank) % local_world_size # form a ring schedule
        for src_token_idx in range(num_init_tokens_per_dev):
            top_k_experts = chosen_experts_cpu[src_dev_idx * num_init_tokens_per_dev + src_token_idx]
            for expert_idx in top_k_experts:
                if expert_start <= expert_idx and expert_idx < expert_end:
                    local_expert_idx = expert_idx % num_experts_per_dev
                    pull_dispatch_indices[token_index_per_expert[local_expert_idx], 0] = src_dev_idx
                    pull_dispatch_indices[token_index_per_expert[local_expert_idx], 1] = src_token_idx
                    token_index_per_expert[local_expert_idx] += 1

    # (Optional) Verify that the dispatch schedule is correct
    start = 0
    for expert_idx in range(expert_start, expert_end):
        token_count = tokens_per_expert[expert_idx]
        padded_token_count = padded_tokens_per_expert[expert_idx]
        end = start + padded_token_count
        assert pull_dispatch_indices[start + token_count:end].sum() == (token_count - padded_token_count) * 2
        assert (pull_dispatch_indices[start:start + token_count, 0] >= 0).all()
        assert (pull_dispatch_indices[start:start + token_count, 1] >= 0).all()
        assert (pull_dispatch_indices[start:start + token_count, 1] < num_init_tokens_per_dev).all()
        assert ((pull_dispatch_indices[start:start + token_count, 0] * num_init_tokens_per_dev + pull_dispatch_indices[start:start + token_count, 1]) < B * S).all()
        assert (pull_dispatch_indices[start:start + token_count, 0] * num_init_tokens_per_dev + pull_dispatch_indices[start:start + token_count, 1]).unique().numel() == token_count
        for token_idx in range(start, (start + token_count).item()):
            src_dev_idx, src_token_idx = pull_dispatch_indices[token_idx]
            assert (chosen_experts[src_dev_idx * num_init_tokens_per_dev + src_token_idx] == expert_idx).any()
        start = end

    # Generate the rest of input/output tensors
    inputs_gathered = torch.zeros(num_padded_local_tokens, H, dtype=torch.bfloat16, device=device)
    inputs_gathered_torch = torch.zeros(num_padded_local_tokens, H, dtype=torch.bfloat16, device=device)
    outputs = torch.zeros(num_padded_local_tokens, I, dtype=torch.bfloat16, device=device)
    outputs_torch = torch.zeros(num_padded_local_tokens, I, dtype=torch.bfloat16, device=device)
    barrier = TKParallelTensor(
        (2, num_padded_max_tokens), # 2 MB is the minimum requirement for multicast object anyways, so no loss here!
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier.data_.zero_()

    torch.distributed.barrier()
    torch.cuda.synchronize()

    torch_run = lambda: torch_moe_dispatch_gemm_func(
        inputs_local=inputs_local,
        inputs_gathered=inputs_gathered_torch, 
        weights=weights,
        outputs=outputs_torch,
        padded_tokens_per_expert=padded_tokens_per_expert,
        pull_dispatch_indices=pull_dispatch_indices,
        local_rank=local_rank,
        local_world_size=local_world_size
    )
    tk_run = lambda: tk_moe_dispatch_gemm_func(
        inputs_local=inputs_local_tk,
        inputs_gathered=inputs_gathered,
        weights=weights,
        outputs=outputs,
        padded_tokens_per_expert=padded_tokens_per_expert,
        pull_dispatch_indices=pull_dispatch_indices,
        barrier=barrier,
        num_comm_sms=num_comm_sms,
        num_padded_local_tokens=num_padded_local_tokens
    )

    if check_correctness:
        clean_print("Checking correctness...", print_once=True)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch_run()
        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("MoE Dispatch GEMM TK vs Torch", outputs, outputs_torch)

    if do_profile:
        clean_print("Profiling...", print_once=True)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        profile(tk_run, num_iters=1)
        torch.distributed.barrier()
        torch.cuda.synchronize()

    tk_avg_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)
    total_flops = 2.0 * (B * S * top_k) * H * I / local_world_size  # this is a rough estimate of per-rank flops
    total_tflops = total_flops * 1e-12
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<MoE Dispatch GEMM | world_size={local_world_size} | num_experts={num_experts} | top_k={top_k} | {B}x{S}x{H}x{I} | num_comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOp/s")


if __name__ == "__main__":
    local_rank, local_world_size = init_distributed_environment()

    # Using DeepSeek V3 model config
    TOP_K = 8
    NUM_EXPERTS = 256
    BATCH_SIZE = 1
    HIDDEN_SIZE = 7168
    EXPERT_HIDDEN_SIZE = 2048

    for seq_len in [8192, 16384, 32768, 65536, 131072]:
        for num_comm_sms in range(28, 54):
            run(
                B=BATCH_SIZE,
                S=seq_len,
                H=HIDDEN_SIZE,
                I=EXPERT_HIDDEN_SIZE,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                num_comm_sms=num_comm_sms,
                local_rank=local_rank,
                local_world_size=local_world_size,
                check_correctness=False,
                do_profile=False
            )

    destroy_distributed_environment()
