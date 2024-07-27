import torch
import sys
import thunderkittens as tk


def make_causal(X):
    (b,n,d) = X.shape
    mask= ~(torch.arange(n).view(1,n,1) >= torch.arange(n).view(1,1,n)).expand(b,n,d)
    X[mask] = 0.
    return X


def based_step_ref(kv_state, k_state, q, k, v, eps=1e-6):
    """
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """
    k_state += k
    kv_state += torch.einsum("bf,bd->bdf", k, v)
    num = torch.einsum("bf,bdf->bd", q, kv_state)
    den = torch.einsum("bf,bf->b", q, k_state) 
    out = num / den.unsqueeze(-1) + eps
    return out, den


def based_test(dt, device='cuda', benchmark=True):
    heads = 1
    itype = dt
    batch_size = 16
    d_model = 64
    d_state = 320

    # prepare inputs 
    torch.random.manual_seed(0)
    kv_state = torch.randn(batch_size*heads, d_model, d_state, dtype=itype, device=device)/d_state
    k_state = torch.randn(batch_size*heads, d_state, dtype=itype, device=device)*5
    v = torch.randn(batch_size*heads, d_model, device=device, dtype=itype)/d_model
    k = torch.randn(batch_size*heads, d_state, device=device, dtype=itype)/d_state
    q = torch.randn(batch_size*heads, d_state, device=device, dtype=itype)/d_state
    kv_state_t = kv_state.transpose(1,2).contiguous() # We prefer the other order for iteration.
        
    # check correctness against reference implementation
    kv_state_ref = kv_state.detach().clone()
    k_state_ref  = k_state.detach().clone()
    out_ref, den_ref    = based_step_ref(kv_state=kv_state_ref, k_state=k_state_ref, v=v, k=k, q=q)
    out_tc       = torch.zeros_like(v)
    tk.based_step(q, k, v, kv_state_t, k_state, out_tc)
    _kv_state = kv_state_t.transpose(1,2) 

    # Overall correctness
    print("Overall correctness:")
    print(f"- out max diff: {(out_tc - out_ref).abs().max().item()}")
    print(f"- k_state  max diff: {(k_state - k_state_ref).abs().max().item()}")
    print(f"- kv_state max diff: {(_kv_state - kv_state_ref).abs().max().item()}")

    print("\nDebugging:")
    has_nans = torch.isnan(out_tc).any()
    has_inf  = torch.isinf(out_tc).any()
    num_infs = torch.isinf(out_tc).sum().item()
    num_nans = torch.isnan(out_tc).sum().item()
    print(f"has_nans: {has_nans}, has_inf: {has_inf}, num_infs: {num_infs}, num_nans: {num_nans}")

    num_diffs = 0
    thresh = 0.001
    # iterate through each batch element and print the diffs between triton and reference
    for i in range(out_ref.shape[0]):
        diff = (out_tc[i] - out_ref[i]).abs().max().item()
        # print(f"{i}: {diff}")
        if diff > thresh: num_diffs += 1
    print(f"{num_diffs=} out of {out_ref.shape[0]} batches, at {thresh=}")


based_test(torch.bfloat16, benchmark=False)
