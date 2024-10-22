import torch
import math
import torch.nn as nn
from einops import rearrange

import thunderkittens as tk


def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol

################ Based Versions ################

eps = 1e-12
TERMS = set([0, 1, 2])

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, head_dim_idx: int = -1, add_scale=False, **kwargs: any):
        super().__init__()
        self.r2  = math.sqrt(2) if add_scale else 1 
        self.rd  = math.sqrt(input_dim) if add_scale else 1 
        self.rrd = math.sqrt(self.rd) if add_scale else 1 
        self.head_dim_idx = head_dim_idx
        
    def forward(self, x: torch.Tensor, chosen_terms = [0, 1, 2]):
        terms_list = []
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.to(torch.float32).unsqueeze(-1) * x.to(torch.float32).unsqueeze(-2)).flatten(start_dim=-2) / (self.r2*self.rd)
        terms = [ x[..., :1] ** 0 ]
        terms.append(x / self.rrd)
        terms.append(x2)
        for i, term in enumerate(terms):
            if i in chosen_terms: terms_list.append(term)
        return torch.cat(terms_list, dim=self.head_dim_idx)


def pytorch_test_v1(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    # SA: note the torch.float32 conversions are very important 

    b, h, n, D = Q.shape
    rd = math.sqrt(D) if add_scale else 1 
    rrd = math.sqrt(rd) if add_scale else 1
    r2 = math.sqrt(2) if add_scale else 1

    print(f"{b=}, {h=}, {n=}, {D=}")
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    # Overall output
    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))
    O2  = make_causal(O.to(torch.float32)**2)
    O1 = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2.to(torch.float32), V.to(torch.float32))
    T2 = T2/(r2 * r2 * rd * rd)
    T1 = torch.einsum("bhnm,bhmd->bhnd", O1.to(torch.float32), V.to(torch.float32))  
    T1 = T1/rd
    T0  = V.to(torch.float32).cumsum(dim=2)

    # KV states by term (a2)
    A2 = torch.einsum("bhnd,bhnf,bhne->bhndef",K.to(torch.float32),V.to(torch.float32),K.to(torch.float32)).cumsum(dim=2) / (r2 * rd) 
    A2 = A2[:, :, -1]
    A2 = rearrange(A2, 'b h e f d -> b h (e f) d')
    K2 = torch.einsum("bhnd,bhne->bhnde", K.to(torch.float32), K.to(torch.float32)) / (rd * r2)
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q.to(torch.float32), Q.to(torch.float32)) / (rd * r2)
    K2 = rearrange(K2, 'b h n d e  -> b h n ( d e )')
    Q2 = rearrange(Q2, 'b h n d e  -> b h n ( d e ) ')
    k_state_a2 = K2.to(torch.float32).cumsum(dim=2)
    D2 = torch.einsum("bhnd,bhnd->bhn", Q2.to(torch.float32), k_state_a2)

    # KV states by term (a1)
    A1 = torch.einsum("bhnd,bhne->bhnde",K.to(torch.float32),V.to(torch.float32)).cumsum(dim=2)  / rrd
    A1 = A1[:, :, -1].transpose(2, 3)
    k_state_a1 = K.to(torch.float32).cumsum(dim=2)/ (rrd)
    D1 = torch.einsum("bhnd,bhnd->bhn", Q.to(torch.float32), k_state_a1) / rrd 

    # KV states by term (a0)
    A0 = V.to(torch.float32).cumsum(dim=2)[:, :, -1]
    K0 = torch.ones(Q[..., :1].shape).to(Q.device)
    D0 =  K0.to(torch.float32).cumsum(dim=2).squeeze(-1)

    numerators   = [T0, T1, T2] 
    denominators = [D0, D1, D2]
    numerator   = sum([n for i, n in enumerate(numerators)   if i in TERMS]) 
    denominator = sum([n for i, n in enumerate(denominators) if i in TERMS]) 
    if add_norm: 
        y = numerator.to(torch.float32) / ( denominator.to(torch.float32).unsqueeze(-1) + eps ).to(torch.bfloat16)
    else:
        y = numerator

    kv_state = torch.cat([A2, A1.transpose(2,3), A0.unsqueeze(-1).transpose(2,3)], dim=2)

    return y, kv_state


def pytorch_test_v2(dt, Q, K, V, d, verbose=True, add_norm=False, add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D, add_scale=add_scale)
    Q = feature_map(Q.to(torch.float32))
    K = feature_map(K.to(torch.float32))
    A_qk = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)) 
    A_qk = torch.tril(A_qk.to(torch.float32))
    y = torch.einsum("bhnm,bhme->bhne", A_qk.to(torch.float32), V.to(torch.float32))

    k_state = K.to(torch.float32).cumsum(dim=2)
    if add_norm:
        den = (Q * k_state).sum(dim=-1) + eps
        y = y / den.unsqueeze(-1)

    return y#, k_state[:,:,-1]


def pytorch_test_v3(dt, Q, K, V, d, verbose=True, add_norm=False,  add_scale=False, **kwargs):
    b, h, n, D = Q.shape
    feature_map = TaylorExp(input_dim=D, add_scale=add_scale)

    # for the output
    q, k = feature_map(Q.to(torch.float32)), feature_map(K.to(torch.float32))
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)
    # overall k_state
    if add_norm:
        denom = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
        out = out / denom

    # for the term 2 kv state (a2)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[2]), feature_map(K.to(torch.float32), chosen_terms=[2])
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A2 = kv_state[:, :, -1].transpose(2, 3)
    D2 = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
    k_state_a2 = k.to(torch.float32).cumsum(dim=2)[:,:,-1].squeeze(-2)

    # for the term 1 kv state (a1)
    q, k = feature_map(Q.to(torch.float32), chosen_terms=[1]), feature_map(K.to(torch.float32), chosen_terms=[1])
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (k.to(torch.float32) * v.to(torch.float32)).cumsum(dim=2)
    A1 = kv_state[:, :, -1]
    D1 = (q.to(torch.float32) * k.to(torch.float32).cumsum(dim=2)).sum(dim=-1) + eps
    k_state_a1 = k.to(torch.float32).cumsum(dim=2)[:,:,-1].squeeze(-2)

    # for the term 0 kv state (a0)
    kv_state = (1 * v.to(torch.float32)).cumsum(dim=2)
    A0 = kv_state[:, :, -1].squeeze(-1)
    K0 = torch.ones(Q[..., :1].to(torch.float32).shape).to(Q.device)
    D0 =  K0.cumsum(dim=2).squeeze(-1)[:,:,-1]

    kv_state = torch.cat([A2, A1.transpose(2,3), A0.unsqueeze(-1).transpose(2,3)], dim=2)

    return out, kv_state


def pytorch_test_v4(dt, Q, K, V, d, verbose=True, add_norm=False,  add_scale=False, **kwargs):
    B, H, L, D = Q.shape

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))**2
    O2  = make_causal(O)
    T2  = torch.einsum("bhnm,bhmd->bhnd", O2.to(torch.float32), V.to(torch.float32)).to(torch.float32)
    T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32)))
    T1 = torch.einsum("bhnm,bhme->bhne", T1a.to(torch.float32), V.to(torch.float32))
    T0  = V.to(torch.float32).cumsum(dim=2)

    rd = math.sqrt(D) if add_scale else 1 
    rrd = math.sqrt(rd) if add_scale else 1
    r2 = math.sqrt(2) if add_scale else 1

    # Denominator
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q.to(torch.float32), Q.to(torch.float32)) / (rd * r2)
    K2 = torch.einsum("bhnd,bhne->bhnde", K.to(torch.float32), K.to(torch.float32)) / (rd * r2)
    k_state_a2 = K2.to(torch.float32).cumsum(dim=2)
    D2 = torch.einsum("bhnde,bhnde->bhn", Q2.to(torch.float32), k_state_a2.to(torch.float32)) 
    k_state_a1 =  K.to(torch.float32).cumsum(dim=2)
    D1 = torch.einsum("bhnd,bhnd->bhn", Q.to(torch.float32), k_state_a1.to(torch.float32))/ ((rrd) ** 2)

    K0 = torch.ones(Q[..., :1].to(torch.float32).shape).to(Q.device)
    D0 =  K0.to(torch.float32).cumsum(dim=2).squeeze(-1)
    
    num, den = [], []
    num.append(T0.to(torch.float32))
    num.append(T1.to(torch.float32) / (rrd * rrd))
    num.append(T2.to(torch.float32) / (rd * r2 * rd * r2))
    if add_norm: 
        den.append(D0.to(torch.float32).unsqueeze(-1))
        den.append(D1.to(torch.float32).unsqueeze(-1))
        den.append(D2.to(torch.float32).unsqueeze(-1))
    
    o = sum([n for i, n in enumerate(num) if i in TERMS])
    den = sum([n for i, n in enumerate(den) if i in TERMS])
    if add_norm: 
        o = o / (den + eps)

    k_state_a2 = rearrange(k_state_a2, 'b h n d e -> b h n (d e)')
    return o #, k_state_a2[:,:,-1], k_state_a1[:,:,-1], D0[:,:,-1]


def based_kernel_test(dt, Q, K, V, d, verbose=True, add_scale=False, add_norm=False, output_state=False):
    b, h, n, d = Q.shape
    dv = V.shape[-1]
    o  = torch.zeros_like(V)

    kv_state_a2 = torch.zeros((b, h, d*d, dv), dtype=dt, device='cuda')
    kv_state_a1 = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    kv_state_a0 = torch.zeros((b, h, dv), dtype=dt, device='cuda')

    o, kv_state = tk.based(
        Q, K, V
    )
    return o, kv_state


################### Benchmarking and Correctness Tests ####################

def linear_attn_correct(dt):
    b = 2
    n = 2048
    h = 2
    head_idx = 1
    d = 16
    dv = 64
    add_scale=True     
    add_norm=False
    output_kv_state=True
    print(f"{b=}, {n=}, {d=}, {h=}")

    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    # get outputs from different methods
    tk_outputs = None 
    fla_parallel_out = None
    pytorch_v1, kv_state_v1  = pytorch_test_v1(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    # pytorch 2 uses a quadratic view so doesn't expose the recurrent state
    pytorch_v2  = pytorch_test_v2(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    pytorch_v3, kv_state_v3 = pytorch_test_v3(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    pytorch_v4 = pytorch_test_v4(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale)
    tk_outputs, kv_state_tk  = based_kernel_test(dt, Q, K, V, d, add_norm=add_norm, add_scale=add_scale, output_state=output_kv_state)
    torch.set_printoptions(sci_mode=False)

    # check overall outputs
    print(f"Note we find numerical differences upon inspecting the tensor outputs:\n")
    print(f"Checking outputs:")
    __eq("PyTorch v1 - PyTorch v2", pytorch_v1, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v2", pytorch_v3, pytorch_v2, debug=False)
    __eq("PyTorch v3 - PyTorch v1", pytorch_v3, pytorch_v1, debug=False)
    __eq("PyTorch v4 - PyTorch v1", pytorch_v4, pytorch_v1, debug=False)
    __eq("PyTorch v4[:,:,:100] - PyTorch v1[:,:,:100]", pytorch_v4[:,:,:100], pytorch_v1[:,:,:100], debug=False)
    if tk_outputs is not None:
        print()
        # __eq("PyTorch v2 - Based TK", pytorch_v2, tk_outputs)
        __eq("PyTorch v1 - Based TK", pytorch_v1, tk_outputs, debug=False)
        __eq("PyTorch v4 - Based TK", pytorch_v4, tk_outputs, debug=False)
        __eq("PyTorch v1[0,0,:15] - Based TK[0,0,:15]", pytorch_v1[0,head_idx,:105], tk_outputs[0,head_idx,:105], debug=False)
        print(pytorch_v4[1,head_idx,0,:4])
        print(tk_outputs[1,head_idx,0,:4])
        print()

    print(pytorch_v4[1,head_idx,70:72,:4])
    print(tk_outputs[1,head_idx,70:72,:4])
    print()

    print(pytorch_v4[0,head_idx,128:130,:4])
    print(tk_outputs[0,head_idx,128:130,:4])
    print()

    print(pytorch_v4[1,head_idx,500:502,:4])
    print(tk_outputs[1,head_idx,500:502,:4])
    print()

    print("---"*10)

    # kv states
    __eq("PyTorch v1 - PyTorch v2", kv_state_v1, kv_state_v3, debug=False)
    __eq("PyTorch v1 - TK", kv_state_v1, kv_state_tk, debug=False)
    __eq("PyTorch v3 - TK", kv_state_v3, kv_state_tk, debug=False)

    print(kv_state_tk[1,head_idx,0,:4])
    print(kv_state_v3[1,head_idx,0,:4])


if __name__ == "__main__":
    linear_attn_correct(torch.bfloat16)