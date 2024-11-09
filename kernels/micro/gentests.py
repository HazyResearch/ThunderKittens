import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 1
N = 16
D = 32

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    x = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == "arange":
    x = torch.arange(B*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, N, D)
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x):
    # TEST 1: add
    o = x + 1

    # TEST 2: warp-multiply
    # o = torch.matmul(x, x.transpose(1, 2))

    return o
o = get_output(x)

with open(f'{TESTNAME}.txt', 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().detach().cpu().numpy().tolist()
    for i in trange(B*N*D):
        f.write(repr(xf[i]))
        f.write(' ')
    for i in trange(B*N*D):
        f.write(repr(of[i]))
        f.write(' ')

