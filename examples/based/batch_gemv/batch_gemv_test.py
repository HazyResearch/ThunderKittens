import torch
import batch_gemv
# import numpy as np

def batch_gemv_ref(input, weight):
    """
    Argument:
        input: (batch, in_features)
        weight: (out_features, in_features)
    Return:
        out: (batch, out_features)
    """
    weight = weight.t().contiguous()
    return input @ weight

def batch_gemv_test(dt, device='cuda', benchmark=True):
    batch_size = 16
    in_features = 4096
    out_features = 14336

    # prepare inputs
    torch.random.manual_seed(0)
    input = torch.rand(batch_size, in_features, device=device, dtype=dt) - 0.5
    weight = torch.rand(out_features, in_features, device=device, dtype=dt) - 0.5
    # input = torch.ones(batch_size, in_features, device=device, dtype=dt) * 0.1
    # weight = torch.ones(out_features, in_features, device=device, dtype=dt) * 0.1
    out_ref = batch_gemv_ref(input, weight)
    out = batch_gemv.batch_gemv(input, weight)

    has_nans = torch.isnan(out).any()
    has_inf  = torch.isinf(out).any()
    num_infs = torch.isinf(out).sum().item()
    num_nans = torch.isnan(out).sum().item()
    print(f"has_nans: {has_nans}, has_inf: {has_inf}, num_infs: {num_infs}, num_nans: {num_nans}")

    # np.savetxt('value.txt', out[0].float().cpu().numpy())
    # np.savetxt('golden_value.txt', out_ref[0].float().cpu().numpy())
    num_diffs = 0
    for i in range(out_ref.shape[0]):
        diff = (out[i] - out_ref[i]).abs().max().item()
        if diff > 1: 
            print(f"{i}: Diff: {diff} Out: {out[i]} Out_ref: {out_ref[i]}")
            num_diffs += 1
            # breakpoint()
    print(f"{num_diffs=}")

    print(f"out max diff: {(out - out_ref).abs().max().item()}")

batch_gemv_test(torch.bfloat16, benchmark=False)