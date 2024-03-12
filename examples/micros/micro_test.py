import torch
import sys
sys.path.append("../../")
from src.common.pyutils.test_build_utils import  __eq
import micro as mod

b = 1
h = 1
n = 16 #*8*2*4
d = 16
dv = 64


def simple_test(dt, use_ones=False):
    (b,h,n,d,dv) = 1,1,128,16,64
    print("Testing at Q,K ({b,h,n,d}) and V ({b,h,n,dv}) ")
    Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
    V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    # call the kernel
    o_small  = torch.zeros_like(Q)
    mod.micro(Q, K, V, o_small)

    o_small_ref = torch.einsum("bhnd,bhnd->bhnd", Q, K)
    __eq("simple dot product (16x16)", o_small_ref.bfloat16(), o_small, debug=False)


profile = False
simple_test(torch.bfloat16)
