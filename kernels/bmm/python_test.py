import torch
import thunderkittens

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
batch_size = 4
M = 128  # weight matrix dimension
K = 64   # inner dimension
max_rank = 16  # maximum LoRA rank

# Create test data
weight = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Create different ranks for each batch
actual_ranks = [4, 8, 16, 0]  # The last one has rank 0 (no adaptation)
rank_tensor = torch.tensor(actual_ranks, dtype=torch.int32, device="cuda")

# Create lora tensors, padded to max_rank
loras = []
for r in actual_ranks:
    # Create a lora tensor with active rows only up to r
    if r > 0:
        lora = torch.randn(r, K, dtype=torch.bfloat16, device="cuda")
        # Pad with zeros to max_rank
        if r < max_rank:
            padding = torch.zeros(max_rank-r, K, dtype=torch.bfloat16, device="cuda")
            lora = torch.cat([lora, padding], dim=0)
    else:
        # Zero lora for rank 0
        lora = torch.zeros(max_rank, K, dtype=torch.bfloat16, device="cuda")
    loras.append(lora)

# Stack all loras
loras = torch.stack(loras, dim=0)  # [B, max_rank, K]

# Run the CUDA kernel
result = thunderkittens.batch_matmul(weight, loras, rank_tensor)

# Verify with PyTorch reference implementation
ref_results = []
for b in range(batch_size):
    r = actual_ranks[b]
    if r > 0:
        # Only calculate for effective rank
        ref_result = torch.matmul(weight, loras[b, :r, :].transpose(0, 1))
        # Pad with zeros if needed
        if r < max_rank:
            padding = torch.zeros(M, max_rank-r, dtype=torch.bfloat16, device="cuda")
            ref_result = torch.cat([ref_result, padding], dim=1)
    else:
        # Zero result for rank 0
        ref_result = torch.zeros(M, max_rank, dtype=torch.bfloat16, device="cuda")
    ref_results.append(ref_result)

ref_results = torch.stack(ref_results, dim=0)  # [B, M, max_rank]

# Check if results match
max_diff = (result - ref_results).abs().max().item()
avg_diff = (result - ref_results).abs().mean().item()

print(f"Max difference: {max_diff}")
print(f"Average difference: {avg_diff}")

# In BFloat16, small differences are expected due to precision
# With matrix multiply, errors can accumulate
tolerance = 1e-2
assert max_diff < tolerance, f"Max difference ({max_diff}) exceeds tolerance ({tolerance})"

print("Test passed!")

# Additional verification: display the effective rank outputs
print("\nVerifying output shapes for each batch:")
for b in range(batch_size):
    r = actual_ranks[b]
    # Count non-zero columns in the output
    nonzero_cols = (result[b].abs().sum(dim=0) > 1e-5).sum().item()
    print(f"Batch {b} (rank {r}): {nonzero_cols} non-zero columns")
    if r > 0:
        assert nonzero_cols == r, f"Expected {r} non-zero columns, got {nonzero_cols}"
