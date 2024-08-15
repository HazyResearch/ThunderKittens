# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch
import einops
idx = 0
from torch.utils.cpp_extension import load
#import attend_micro as mod
debug=False
import sys
sys.path.append('../')
from util import __eq

import thunderkittens as tk

# dtypes = [torch.bfloat16, torch.float16, torch.float]
dtypes = [torch.bfloat16]

### TILE TESTS
b = 16
h = 16
n = 1024 # 1024 # 8*16*2*2 #*2*2
d = 64
nWarps   = 8
tile_dim = 16
tile_size = tile_dim*tile_dim

#def _rtile(dtype): return torch.ones(b,h,n,d,device='cuda',dtype=dtype)
def _rtile(dtype): return torch.randn(b,h,n,d,device='cuda',dtype=dtype)/(n*d)
def _zero(x): return torch.zeros(*x.shape, dtype=x.dtype, device=x.device)
def _ones(s): return torch.ones(*s, dtype=dt, device='cuda')
def _rand(s):  return torch.randn(*s, dtype=dt, device='cuda')*1e-1

def sliding_window(dt, use_ones=True):
    s = (b,h,n,d)
    _gen = _ones if use_ones else _rand
    j     = 85
    q,k,v = tuple(_gen(s) for _ in range(3))
    print(q)
    o     = torch.zeros(d, dtype=dt, device='cuda')
    tk.based_sliding_window(q, k, v, o)
    window_size = 256
    kw = k[:,:,j-window_size:j]
    vw = v[:,:,j-window_size:j]

    w = torch.einsum("bhd, bhnd-> bhn",q[:,:,j].float(),kw.float())
    a = torch.nn.functional.softmax(w, dim=-1)
    o0 = torch.einsum("bhn,bhnd->bhd", a, vw.float())
    __eq("sliding window", o0.bfloat16(), o, debug=debug)

tests = [sliding_window]

banner="**************"
for dt in dtypes:
    print(f"{banner}\nStarting {dt}\n{banner}")
    for t in tests: t(dt)