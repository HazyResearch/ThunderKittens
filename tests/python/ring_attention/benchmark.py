from functools import partial
import gc

import jax
import numpy as np
import torch

from implementations import (
    generate_mha_inputs,
    mha_torch,
    mha_jax,
    ring_mha_orig
)

from tk_ring_attention import ring_mha_forward, ring_mha_backward, pgl_tensor

# Parameters
NUM_DEVICES = 8
B = 16 # batch size
H = 16 # number of heads
N = 4096 * NUM_DEVICES # sequence length
D_h = 64 # head dimension
dtype = 'bf16'
causal = False

# Generate inputs
Qs, Ks, Vs, dLs = generate_mha_inputs(
    B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES
)

print(type(Qs))
print(type(Qs[0]))

ring_Os = ring_mha_forward(Qs, Ks, Vs, causal)
normal_Os = []

for i in range(NUM_DEVICES):
    torch.cuda.set_device(i)
    normal_Os.append(
        mha_torch(Qs[i], Ks[i], Vs[i], causal)
    )
    torch.cuda.synchronize(i)

TOL = 1e-2 # large due to bf16
for i in range(NUM_DEVICES):
    print(f'Device {i}:')
    assert(normal_Os[i].shape == ring_Os[i].shape)
    assert(normal_Os[i].dtype == ring_Os[i].dtype)
    assert(normal_Os[i].device == ring_Os[i].device)
    normal_O = normal_Os[i].detach().to(dtype=torch.float32, device='cpu').numpy()
    ring_O = ring_Os[i].detach().to(dtype=torch.float32, device='cpu').numpy()
    max_error = np.max(np.abs(normal_O - ring_O))
    num_errors = np.sum(np.abs(normal_O - ring_O) > TOL)
    print(f'  Max abs diff: {max_error}')
    print(f'  Num errors: {num_errors} out of {normal_O.size} (TOL: {TOL})')
