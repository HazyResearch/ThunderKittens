import torch
import sys

B = 1
H = 1 
D = 64
DV = 320

TESTNAME = sys.argv[1]

if TESTNAME in ['ones']:
    kv_state = torch.ones(B * H, D, DV, dtype=torch.bfloat16, device='cuda') / DV
    k_state  = torch.ones(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    v        = torch.ones(B * H, D,     dtype=torch.bfloat16, device='cuda') / D
    k        = torch.ones(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    q        = torch.ones(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    
elif TESTNAME in ['randn']:
    torch.random.manual_seed(42)
    
    kv_state = torch.randn(B * H, D, DV, dtype=torch.bfloat16, device='cuda') / DV
    k_state  = torch.randn(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    v        = torch.randn(B * H, D,     dtype=torch.bfloat16, device='cuda') / D
    k        = torch.randn(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    q        = torch.randn(B * H, DV,    dtype=torch.bfloat16, device='cuda') / DV
    
else:
    print('Invalid test name')
    sys.exit(0)


def pytorch_test(kv_state, k_state, q, k, v, TESTNAME='all'):
    
    def make_causal(X):
        (b,n,d) = X.shape
        mask= ~(torch.arange(n).view(1,n,1) >= torch.arange(n).view(1,1,n)).expand(b,n,d)
        X[mask] = 0.
        return X
    
    kv_state_ref = kv_state.detach().clone()
    k_state_ref  = k_state.detach().clone()

    k_state += k
    kv_state += torch.einsum("bf,bd->bdf", k, v)
    num = torch.einsum("bf,bdf->bd", q, kv_state)
    den = torch.einsum("bf,bf->b", q, k_state) + 1e-6
    
    out_ref = num / den.unsqueeze(-1)
    
    return out_ref, kv_state_ref, k_state_ref

o_ref, kv_state_ref, k_state_ref = pytorch_test(kv_state, k_state, q, k, v, TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    
    kv_state_f = kv_state_ref.to(torch.float32).flatten().cpu().numpy()
    k_state_f  = k_state_ref.to(torch.float32).flatten().cpu().numpy()
    of = o_ref.to(torch.float32).flatten().cpu().numpy()
    
    for i in range(B * H * DV):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in range(B * H * DV):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in range(B * H * D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in range(B * H * D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in range(B * H * DV):
        f.write(repr(k_state_f[i]))
        f.write(' ')
    for i in range(B * H * DV * D):
        f.write(repr(kv_state_f[i]))
        f.write(' ')
