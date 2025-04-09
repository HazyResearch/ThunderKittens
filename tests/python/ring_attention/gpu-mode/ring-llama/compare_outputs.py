import torch

a = torch.load("original_attn_output.pt")
b = torch.load("ring_attn_output.pt")

abs_delta = torch.abs(a - b)

print(f"dtype={abs_delta.dtype}")
print(f"total delta: {abs_delta.sum().item()} (numel={abs_delta.numel()}), mean={abs_delta.mean().item():.4f}, max={abs_delta.max().item():.4f}")

mean_abs_error_per_row = abs_delta.mean(dim=-1)
max_abs_error_per_row,_ = abs_delta.max(dim=-1)

#print("delta_per_row:", delta_per_row)
for i, (mean_err, max_err) in enumerate(zip(mean_abs_error_per_row[0], max_abs_error_per_row[0])):
    print(f"error q[{i}]: mean_err={mean_err.item():.4f}, max_err={max_err.item():.4f}")
