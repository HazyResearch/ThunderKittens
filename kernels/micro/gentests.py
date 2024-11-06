import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math

B = 1
N = 16
D = 16

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    x = torch.zeros((B, N, D), dtype=torch.bfloat16, device='cuda')
else:
    print('Invalid test name')
    sys.exit(0)

def get_output(x):
    o = x + 1
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

