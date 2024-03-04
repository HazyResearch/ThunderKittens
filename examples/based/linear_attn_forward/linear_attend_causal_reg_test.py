import torch

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)
from src.pyutils.test_build_utils import jit_build, print_tiles, _rtile, __eq
idx = 0
debug = False
tile  = 16
prebuilt= True
if prebuilt:
    import linear_attend_causal_reg as mod
else:
    mod = jit_build("linear_attend_causal_reg")

def make_causal(X):
    (b,n,d) = X.shape
    mask= ~(torch.arange(n).view(1,n,1) >= torch.arange(n).view(1,1,n)).expand(b,n,d)
    X[mask] = 0.
    return X


b = 16
h = 16
b = 1
h = 1
n = 16*8*2*4
d = 16
dv = 64

# b = 1
# h = 1
# n = 16*8*4
# d = 16

def a2_test(dt):
    s = (b,h,n,d)
    Q,K = [torch.ones(b,h,n,d, dtype=dt, device='cuda') for _ in range(2)]
    V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')
    # Q,K,V
    Q,K = [torch.randn(b,h,n,d, dtype=dt, device='cuda')/d for _ in range(2)]
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv
    
    A2 = torch.einsum("bhnd,bhne,bhnf->bhndef",K,K,V).cumsum(dim=2) 
    Q2 = torch.einsum("bhnd,bhne->bhnde", Q, Q) 
    T2 = torch.einsum("bhnde,bhndef->bhnf", Q2, A2) 
    o    = torch.zeros(*V.shape, dtype=dt, device='cuda')
    mod.a2_compute(Q,K,V,o)
    __eq("a2_test", T2,o, debug=False)
    print(f"T2\n{T2}")
    print(f"o\n{o}")
    delta = o-T2
    print(f"diff\n{delta}")
    print(f"max={delta.abs().max()}")
    print(f"{delta.shape}")
    for i in range(n//16): print(f"o{i}={o[0,0,i*16:(i+1)*16,0]}")

    print("*** T2 ***")
    for i in range(n//16): print(f"Tw{i}={T2[0,0,i*16:(i+1)*16,0]}")


a2_test(torch.bfloat16)
