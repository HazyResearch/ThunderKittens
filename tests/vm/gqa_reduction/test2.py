import math
import time  # Keep for reference implementation timing if desired

import torch

# No need to explicitly import torch.cuda.Event, it's part of torch.cuda
from gqa_reduction import gqa_reduction

# Fixed VM parameters
NUM_BLOCKS_REDUCTION = 32  # Must match NUM_Q_HEADS for the reduction kernel
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_REDUCTION_OPCODE = 2

# Model/Attention Parameters (match kernel constants)
H_q = 32  # number of query heads (NUM_Q_HEADS in CUDA)
D_h = 64  # dimension of each head (HEAD_DIM in CUDA)

# Test Configuration
NUM_PARTIALS = 8  # Number of partial results to reduce (must be <= MAX_PARTIALS)
MAX_PARTIALS = 1024  # Max intermediate partials storage capacity (M_a)

torch.manual_seed(0)

# --- Helper Functions ---

device = "cuda:0"


def generate_inputs(
    num_q_heads: int, head_dim: int, num_partials: int, max_partials_storage: int
):
    # LSE Partials Input: Shape (H_q, M_a), dtype float32
    LSE_partials_in = torch.randn(
        (num_q_heads, max_partials_storage),
        dtype=torch.float32,
        device=device,
    )

    # O Partials Input: Shape (H_q, M_a, D_h), dtype float32
    O_partials_in = torch.randn(
        (num_q_heads, max_partials_storage, head_dim),
        dtype=torch.float32,
        device=device,
    )

    # Final Output Tensor: Shape (H_q, D_h), dtype bfloat16
    O_final_out = torch.zeros(
        num_q_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    return LSE_partials_in, O_partials_in, O_final_out


def generate_instructions_and_timings_reduction(
    num_q_heads_to_run: int, num_partials_total: int
):
    """Generates instructions for the reduction kernel."""
    instructions = []
    max_instructions = 1  # Only one reduction instruction per Q head block

    # Assign one reduction instruction per Q head block
    for q_idx in range(num_q_heads_to_run):
        inst = [GQA_REDUCTION_OPCODE, num_partials_total] + [0] * (
            INSTRUCTION_WIDTH - 2
        )
        instructions.append([inst])

    # Pad instructions if NUM_Q_HEADS < NUM_BLOCKS_REDUCTION
    while len(instructions) < NUM_BLOCKS_REDUCTION:
        instructions.append(
            [[0] * INSTRUCTION_WIDTH for _ in range(max_instructions)]
        )  # Add padding instruction lists

    instructions_tensor = torch.tensor(instructions, dtype=torch.int32).to(
        device=device
    )
    timings_tensor = torch.zeros(
        (NUM_BLOCKS_REDUCTION, max_instructions, TIMING_WIDTH), dtype=torch.int32
    ).to(device=device)

    return instructions_tensor, timings_tensor


def reference_attention_reduction(
    L_partials: torch.Tensor,  # Shape: (H_q, M_a)
    O_partials: torch.Tensor,  # Shape: (H_q, M_a, D_h)
    num_partials_to_reduce: int,
    num_q_heads: int,
    head_dim: int,
) -> torch.Tensor:
    O_final_ref = torch.zeros(
        num_q_heads, head_dim, dtype=torch.float32, device=L_partials.device
    )

    for head_idx in range(num_q_heads):
        lses = L_partials[head_idx, :num_partials_to_reduce].float()
        outs = O_partials[head_idx, :num_partials_to_reduce, :].float()

        max_lse = torch.max(lses)
        adjusted_factors = torch.exp2(lses - max_lse)
        new_denominator = adjusted_factors.sum()

        reduced = (outs * adjusted_factors.unsqueeze(1)).sum(dim=0) / new_denominator
        O_final_ref[head_idx] = reduced

    return O_final_ref


# --- Main Execution ---

LSE_partials_py, O_partials_py, O_final_py = generate_inputs(
    H_q, D_h, NUM_PARTIALS, MAX_PARTIALS
)

LSE_partials_kernel = LSE_partials_py.unsqueeze(0).unsqueeze(0)
O_partials_kernel = O_partials_py.unsqueeze(0)
O_final_kernel = O_final_py.unsqueeze(0).unsqueeze(2)

reduction_instructions, reduction_timings = generate_instructions_and_timings_reduction(
    H_q, NUM_PARTIALS
)

gqa_reduction(
    reduction_instructions,
    reduction_timings,
    LSE_partials_kernel,
    O_partials_kernel,
    O_final_kernel,
)
torch.cuda.synchronize()

# Timing using CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

num_iters = 0
if num_iters > 0:
    start_event.record()
    for _ in range(num_iters):
        gqa_reduction(
            reduction_instructions,
            reduction_timings,
            LSE_partials_kernel,
            O_partials_kernel,
            O_final_kernel,
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    time_per_iter_us = (elapsed_time_ms * 1e3) / num_iters
    print(f"Time per iter (reduction): {time_per_iter_us:.2f} us")


# Squeeze the output tensor back to the standard Python shape for comparison
# (1, H_q, 1, D_h) -> (H_q, D_h)
O_final_squeezed = O_final_kernel.squeeze(2).squeeze(0)

# Convert final CUDA output (bf16) to float32 for comparison
O_final_cuda = O_final_squeezed.float()

O_final_ref = reference_attention_reduction(
    LSE_partials_py, O_partials_py, NUM_PARTIALS, H_q, D_h
)

diff = O_final_cuda - O_final_ref
adiff = torch.abs(diff)
rdiff = 2 * adiff / (torch.abs(O_final_cuda) + torch.abs(O_final_ref))

print(f"adiff: mean={adiff.mean()}, max={adiff.max()}")
print(f"rdiff: mean={rdiff.mean()}, max={rdiff.max()}")
