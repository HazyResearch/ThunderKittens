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

from tk_ring_attention import ring_mha_forward, ring_mha_backward

# Parameters
NUM_DEVICES = 8
B = 8 # batch size
H = 16 # number of heads
N = 1024 # sequence length
D_h = 32 # head dimension
dtype = 'bf16'
causal = False

# Generate inputs
Qs, Ks, Vs, dLs = generate_mha_inputs(
    B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES
)

print(type(Qs))
print(type(Qs[0]))

out = ring_mha_forward(Qs, Ks, Vs, causal)