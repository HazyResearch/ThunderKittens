import torch
import triton
import triton.language as tl

def hedgehog_step(kv_state, k_state, q, k, v, denom: bool=False):
    """
    Borrows from Mamba implementation
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """
    batch, dim, d_state = kv_state.shape
    assert v.shape == (batch, dim)
    assert q.shape == (batch, d_state)
    assert k.shape == q.shape
    out = torch.empty_like(v)
   
    # This is currently not blocking along the d_state dimension which 
    # is okay for small d_state (e.g. 16 in Mamba), but an issue for the state sizes
    # we're dealing with 
    BLOCK_SIZE_M, num_warps = (16,16)
    BLOCK_SIZE_DSTATE = d_state
    grid = lambda META: (triton.cdiv(dim, BLOCK_SIZE_M), batch)

    # print(f"Triton: ")
    # print(f"-- {out.stride(0)=}, {out.stride(1)=}")
    # print(f"-- {kv_state.stride(0)=}, {kv_state.stride(1)=}, {kv_state.stride(2)=}")
    # print(f"-- {k_state.stride(0)=}, {k_state.stride(1)=}")
    # print(f"-- {v.stride(0)=}, {v.stride(1)=}")
    # print(f"-- {k.stride(0)=}, {k.stride(1)=}")
    # print(f"-- {q.stride(0)=}, {q.stride(1)=}")

    with torch.cuda.device(v.device.index):
        _hedgehog_step[grid](
            kv_state, k_state, v, k, q, out,
            dim, d_state,
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
            k_state.stride(0), k_state.stride(1),
            v.stride(0), v.stride(1),
            k.stride(0), k.stride(1),
            q.stride(0), q.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_DSTATE,
            num_warps=num_warps,
        )
    return out


@triton.jit
def _hedgehog_step(
    # Pointers to matrices
    kv_state_ptr, k_state_ptr, V_ptr, K_ptr, Q_ptr, out_ptr, 
    # Matrix dimensions
    dim, dstate,
    # Strides
    stride_kv_state_batch, stride_kv_state_dim, stride_kv_state_dstate,
    stride_k_state_batch, stride_k_state_dstate,
    stride_v_batch, stride_v_dim,
    stride_k_batch, stride_k_dstate,
    stride_q_batch, stride_q_dstate,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    kv_state_ptr += pid_b * stride_kv_state_batch
    k_state_ptr += pid_b * stride_k_state_batch
    V_ptr += pid_b * stride_v_batch
    K_ptr += pid_b * stride_k_batch
    Q_ptr += pid_b * stride_q_batch
    out_ptr += pid_b * stride_out_batch

    # set pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    kv_state_ptrs = kv_state_ptr + (offs_m[:, None] * stride_kv_state_dim + offs_n[None, :] * stride_kv_state_dstate)
    k_state_ptrs = k_state_ptr + offs_n * stride_k_state_dstate
    v_ptrs = V_ptr + offs_m * stride_v_dim
    k_ptrs = K_ptr + offs_n * stride_k_dstate
    q_ptrs = Q_ptr + offs_n * stride_q_dstate
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # loads
    kv_state = tl.load(kv_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    k_state = tl.load(k_state_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    V = tl.load(v_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    K = tl.load(k_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    Q = tl.load(q_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

    # compute state updates
    kv_state = kv_state + K[None, :] * V[:, None]
    tl.store(kv_state_ptrs, kv_state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    k_state = k_state + K  
    tl.store(k_state_ptrs, k_state, mask=(offs_n < dstate))
    
    # compute output
    num = tl.sum(kv_state * Q[None, :], axis=1)
    den = tl.sum(k_state  * Q[None, :], axis=1) + 1e-6
    out = num / den

    tl.store(out_ptrs, out, mask=offs_m < dim)


def hedgehog_step_ref(kv_state, k_state, q, k, v, denom: bool=True, eps: float=1e-6):
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
    den = torch.einsum("bf,bf->b", q, k_state) + eps
    return num / den.unsqueeze(-1) #, den


def hedgehog_step_tk(kv_state, kv_state_t, k_state, q, k, v, denom):
    out_tk       = torch.zeros_like(v)
    import based_inference as mod
    mod.based_step(q, k, v, kv_state_t, k_state, out_tk, denom)
    return out_tk


def hedgehog_test(dt, device='cuda', benchmark=True, ones=False):
    itype = dt
    batch_size = 32
    d_model = 64
    d_state = 256

    print(f"Running hedgehog test with {dt} -- benchmark={benchmark} -- ones={ones}")

    # prepare inputs 
    torch.random.manual_seed(0)
    if not ones:
        kv_state = torch.randn(batch_size, d_model, d_state, dtype=itype, device=device)/d_state
        k_state = torch.randn(batch_size, d_state, dtype=itype, device=device)
        v = torch.randn(batch_size, d_model, device=device, dtype=itype)/d_model
        k = torch.randn(batch_size, d_state, device=device, dtype=itype)/d_state
        q = torch.randn(batch_size, d_state, device=device, dtype=itype)/d_state
    else:
        kv_state = torch.ones(batch_size, d_model, d_state, dtype=itype, device=device)
        k_state = torch.ones(batch_size, d_state, dtype=itype, device=device)
        v = torch.ones(batch_size, d_model, device=device, dtype=itype)
        k = torch.ones(batch_size, d_state, device=device, dtype=itype)
        q = torch.ones(batch_size, d_state, device=device, dtype=itype)
        
    # check correctness against reference implementation
    kv_state_ref  = kv_state.detach().clone()
    k_state_ref   = k_state.detach().clone()
    kv_state_tri  = kv_state.detach().clone()
    k_state_tri   = k_state.detach().clone()
    kv_state_tk   = kv_state.detach().clone()
    k_state_tk    = k_state.detach().clone()

    outputs = []
    out_ref       = hedgehog_step_ref(kv_state=kv_state_ref, k_state=k_state_ref, v=v, k=k, q=q, denom=True)
    out_triton    = hedgehog_step(kv_state=kv_state_tri, k_state=k_state_tri, v=v, k=k, q=q, denom=True)
    outputs.append((out_triton, kv_state_tri, k_state_tri, "Triton"))
    
    if dt != torch.float32:
        denom = torch.zeros(batch_size, 1, device=device, dtype=itype)/d_model
        kv_state_t = kv_state_tk.transpose(1,2).contiguous()
        out_tk = hedgehog_step_tk(kv_state_tk, kv_state_t, k_state_tk, q, k, v, denom)
        outputs.append((out_tk, kv_state_tk, k_state_tk, "ThunderKittens"))

    for (out, kv_state, k_state, name) in outputs:
        print(f"**** {name} dtype={dt} ****")
        print(f"out max diff: {(out - out_ref).abs().max().item()}")
        print(f"k_state  max diff: {(k_state - k_state_ref).abs().max().item()}")
        print(f"kv_state max diff: {(kv_state - kv_state_ref).abs().max().item()}")


if __name__ == "__main__":
    hedgehog_test(torch.bfloat16, benchmark=False, ones=False)
    hedgehog_test(torch.bfloat16, benchmark=False, ones=True)
    hedgehog_test(torch.float32, benchmark=False, ones=False) # fp32 not supported for TK, only Triton
    hedgehog_test(torch.float32, benchmark=False, ones=True)