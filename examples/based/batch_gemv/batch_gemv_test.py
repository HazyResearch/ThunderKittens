import torch
import batch_gemv

def batch_gemv_ref(input, weight):
    """
    Argument:
        input: (batch, in_features)
        weight: (out_features, in_features)
    Return:
        out: (batch, out_features)
    """
    return input @ weight.t()

def batch_gemv_test(dt, device='cuda', benchmark=True):
    batch_size = 16
    in_features = 4096
    out_features = 14336

    # prepare inputs
    torch.random.manual_seed(0)
    input = torch.randn(batch_size, in_features, device=device, dtype=dt)
    weight = torch.randn(out_features, in_features, device=device, dtype=dt)
    out_ref = batch_gemv_ref(input, weight)
    out = batch_gemv.batch_gemv(input, weight)

    has_nans = torch.isnan(out).any()
    has_inf  = torch.isinf(out).any()
    num_infs = torch.isinf(out).sum().item()
    num_nans = torch.isnan(out).sum().item()
    print(f"has_nans: {has_nans}, has_inf: {has_inf}, num_infs: {num_infs}, num_nans: {num_nans}")

    num_diffs = 0
    for i in range(out_ref.shape[0]):
        diff = (out[i] - out_ref[i]).abs().max().item()
        if diff > 1: 
            print(f"{i}: {diff}")
            num_diffs += 1
            # breakpoint()
    print(f"{num_diffs=}")

    print(f"out max diff: {(out - out_ref).abs().max().item()}")

batch_gemv_test(torch.bfloat16, benchmark=False)